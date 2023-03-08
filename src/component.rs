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
    pub(crate) index: usize,
    //generation: usize,
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

    pub(crate) fn id(&self) -> ComponentID {
        self.ty
    }
    pub(crate) fn index(&self) -> usize {
        self.index
    }
}

/// identifier for a comp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentID {
    pub(crate) name: &'static str,
    pub(crate) id: TypeId,
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
pub trait Component: Clone + 'static {
    fn id() -> ComponentID {
        ComponentID {
            name: type_name::<Self>(),
            id: TypeId::of::<Self>(),
        }
    }

    fn id_instance(&self) -> ComponentID {
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
