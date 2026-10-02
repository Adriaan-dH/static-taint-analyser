"""Immutable values, variable bindings, and allocation-site list memory."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AbstractValue:
    """Scalar possibilities and references can coexist after a path join."""

    scalar_tainted: bool = False
    list_objects: frozenset[int] = frozenset()
    may_be_scalar: bool = False

    def join(self, other: "AbstractValue") -> "AbstractValue":
        return AbstractValue(
            self.scalar_tainted or other.scalar_tainted,
            self.list_objects | other.list_objects,
            self.may_be_scalar or other.may_be_scalar,
        )


CLEAN_SCALAR = AbstractValue(may_be_scalar=True)
TAINTED_SCALAR = AbstractValue(scalar_tainted=True, may_be_scalar=True)


@dataclass(frozen=True)
class ListMemory:
    """Exact cells plus a summary for writes to an unknown index.

    Exact cells already include earlier unknown writes. This allows a later
    strong overwrite to kill taint at one cell without clearing other cells.
    Bounds do not prune control flow: even an out-of-range write is represented
    conservatively, rather than modelling Python exceptions.
    """

    length: int
    elements: tuple[tuple[int, AbstractValue], ...] = ()
    unknown: AbstractValue = AbstractValue()
    multiple_allocations: bool = False

    def normalise_index(self, index: int) -> int:
        return self.length + index if -self.length <= index < 0 else index

    def read(self, index: int | None) -> AbstractValue:
        if index is not None:
            return dict(self.elements).get(self.normalise_index(index), self.unknown)
        result = self.unknown
        for _, value in self.elements:
            result = result.join(value)
        return result

    def write(self, index: int | None, value: AbstractValue, strong: bool) -> "ListMemory":
        elements = dict(self.elements)
        if index is None:
            elements = {key: old.join(value) for key, old in elements.items()}
            return replace(
                self, elements=tuple(sorted(elements.items())), unknown=self.unknown.join(value)
            )
        index = self.normalise_index(index)
        elements[index] = value if strong else self.read(index).join(value)
        return replace(self, elements=tuple(sorted(elements.items())))

    def join(self, other: "ListMemory") -> "ListMemory":
        if self.length != other.length:
            raise ValueError("One list allocation site must have a fixed literal length")
        indices = dict(self.elements).keys() | dict(other.elements).keys()
        return ListMemory(
            self.length,
            tuple((index, self.read(index).join(other.read(index))) for index in sorted(indices)),
            self.unknown.join(other.unknown),
            self.multiple_allocations or other.multiple_allocations,
        )


@dataclass(frozen=True)
class TaintState:
    """Sorted immutable maps keep stored worklist states comparable and stable."""

    bindings: tuple[tuple[str, AbstractValue], ...] = ()
    lists: tuple[tuple[int, ListMemory], ...] = ()

    @property
    def tainted_variables(self) -> frozenset[str]:
        return frozenset(name for name, value in self.bindings if value.scalar_tainted)

    def value(self, variable: str) -> AbstractValue:
        return dict(self.bindings).get(variable, AbstractValue())

    def list_memory(self, object_id: int) -> ListMemory:
        return dict(self.lists)[object_id]

    def bind(self, variable: str, value: AbstractValue) -> "TaintState":
        bindings = dict(self.bindings)
        bindings[variable] = value
        return replace(self, bindings=tuple(sorted(bindings.items())))

    def with_list(self, object_id: int, memory: ListMemory) -> "TaintState":
        lists = dict(self.lists)
        lists[object_id] = memory
        return replace(self, lists=tuple(sorted(lists.items())))

    def allocate(self, object_id: int, elements: tuple[AbstractValue, ...]) -> "TaintState":
        memory = ListMemory(len(elements), tuple(enumerate(elements)))
        if object_id in dict(self.lists):
            # An incoming object at its allocation point can be an older runtime
            # instance from a back-edge. Preserve it and prohibit strong writes.
            memory = replace(self.list_memory(object_id).join(memory), multiple_allocations=True)
        return self.with_list(object_id, memory)

    def value_is_tainted(self, value: AbstractValue) -> bool:
        """A whole-list sink observes any reachable tainted element, cycle-safely."""
        pending = [value]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current.scalar_tainted:
                return True
            for object_id in current.list_objects - visited:
                visited.add(object_id)
                memory = self.list_memory(object_id)
                pending.append(memory.unknown)
                pending.extend(element for _, element in memory.elements)
        return False

    def is_tainted(self, variable: str) -> bool:
        return self.value_is_tainted(self.value(variable))

    def taint(self, variable: str) -> "TaintState":
        return self.bind(variable, TAINTED_SCALAR)

    def clean(self, variable: str) -> "TaintState":
        if variable not in dict(self.bindings):
            return self
        return self.bind(variable, CLEAN_SCALAR)

    def join(self, other: "TaintState") -> "TaintState":
        """Union values and shared-object contents from either incoming path."""
        bindings = dict(self.bindings)
        for name, value in other.bindings:
            bindings[name] = bindings.get(name, AbstractValue()).join(value)
        lists = dict(self.lists)
        for object_id, memory in other.lists:
            lists[object_id] = lists[object_id].join(memory) if object_id in lists else memory
        return TaintState(tuple(sorted(bindings.items())), tuple(sorted(lists.items())))
