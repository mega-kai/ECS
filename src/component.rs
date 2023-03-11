use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    fmt::Debug,
    mem::size_of,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentKey {
    pub(crate) index: usize,
    pub(crate) ty: ComponentID,
}
impl ComponentKey {
    pub(crate) fn new<C: Component>(index: usize) -> Self {
        Self {
            index,
            //generation: 0,
            ty: ComponentID::new::<C>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentID {
    pub(crate) column_index: usize,
    pub(crate) name: &'static str,
    pub(crate) id: TypeId,
}
impl ComponentID {
    pub(crate) fn new<C: Component>() -> Self {
        Self {
            name: type_name::<C>(),
            id: TypeId::of::<C>(),
            column_index: 0,
        }
    }
}

pub trait Component: Clone + 'static {
    fn id() -> ComponentID {
        ComponentID {
            name: type_name::<Self>(),
            id: TypeId::of::<Self>(),
            column_index: 0,
        }
    }

    fn id_instance(&self) -> ComponentID {
        ComponentID {
            name: type_name::<Self>(),
            id: TypeId::of::<Self>(),
            column_index: 0,
        }
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }

    fn size(&self) -> usize {
        size_of::<Self>()
    }
}

impl<C0: Component> Component for (C0,) {}
impl<C0: Component, C1: Component> Component for (C0, C1) {}
impl<C0: Component, C1: Component, C2: Component> Component for (C0, C1, C2) {}
impl<C0: Component, C1: Component, C2: Component, C3: Component> Component for (C0, C1, C2, C3) {}
impl<C0: Component, C1: Component, C2: Component, C3: Component, C4: Component> Component
    for (C0, C1, C2, C3, C4)
{
}
impl<C0: Component, C1: Component, C2: Component, C3: Component, C4: Component, C5: Component>
    Component for (C0, C1, C2, C3, C4, C5)
{
}
impl<
        C0: Component,
        C1: Component,
        C2: Component,
        C3: Component,
        C4: Component,
        C5: Component,
        C6: Component,
    > Component for (C0, C1, C2, C3, C4, C5, C6)
{
}
impl<
        C0: Component,
        C1: Component,
        C2: Component,
        C3: Component,
        C4: Component,
        C5: Component,
        C6: Component,
        C7: Component,
    > Component for (C0, C1, C2, C3, C4, C5, C6, C7)
{
}
