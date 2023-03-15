use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    fmt::Debug,
    mem::size_of,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentAccess {
    pub(crate) entity_id: usize,
    pub(crate) id: ComponentID,
    pub(crate) access: *mut u8,
}
impl ComponentAccess {
    pub(crate) fn new(entity_id: usize, ty: ComponentID, access: *mut u8) -> Self {
        Self {
            entity_id,
            id: ty,
            access,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentID {
    // for debugging
    pub(crate) name: &'static str,
    pub(crate) type_id: TypeId,
}
impl ComponentID {
    pub(crate) fn new<C: Component>() -> Self {
        Self {
            name: type_name::<C>(),
            type_id: TypeId::of::<C>(),
        }
    }
}

pub trait Component: Clone + 'static {
    fn id() -> ComponentID {
        ComponentID {
            name: type_name::<Self>(),
            type_id: TypeId::of::<Self>(),
        }
    }

    fn id_instance(&self) -> ComponentID {
        ComponentID {
            name: type_name::<Self>(),
            type_id: TypeId::of::<Self>(),
        }
    }

    // fn layout() -> Layout {
    //     Layout::new::<Self>()
    // }

    // fn size() -> usize {
    //     size_of::<Self>()
    // }
}

impl<C0: Component> Component for (C0,) {}
impl<C0: Component, C1: Component> Component for (C0, C1) {}
impl<C0: Component, C1: Component, C2: Component> Component for (C0, C1, C2) {}
impl<C0: Component, C1: Component, C2: Component, C3: Component> Component for (C0, C1, C2, C3) {}
