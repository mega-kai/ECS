//component its related items
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    fmt::Debug,
    mem::size_of,
};

/// a key to access a component, which includes entities and child entities
#[derive(Debug, Clone, Copy)]
pub struct ComponentKey {
    index: usize,
    generation: usize,
    ty: ComponentID,
}
impl ComponentKey {
    pub(crate) fn new<C: Component>(index: usize) -> Self {
        Self {
            index: index,
            generation: 0,
            ty: ComponentID::new::<C>(),
        }
    }
}

/// ID used for comparing component types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentID {
    name: &'static str,
    id: TypeId,
}
impl ComponentID {
    pub(crate) fn new<C: Component>() -> Self {
        Self {
            name: type_name::<C>(),
            id: TypeId::of::<C>(),
        }
    }
}

/// marker trait for components
/// TODO! proc macro derive
pub trait Component: Copy + Clone + 'static {
    fn id(&self) -> ComponentID {
        ComponentID {
            name: type_name::<Self>(),
            id: TypeId::of::<Self>(),
        }
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }

    fn size(&self) -> usize {
        size_of::<Self>()
    }
}
