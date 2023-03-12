use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    fmt::Debug,
    mem::size_of,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentAccess {
    pub(crate) row_index: usize,
    pub(crate) ty: ComponentID,
    pub(crate) access: *mut u8,
}
impl ComponentAccess {
    pub(crate) fn new_from_type<C: Component>(index: usize) -> Self {
        Self {
            row_index: index,
            //generation: usize,
            ty: ComponentID::new::<C>(),
            access: Layout::new::<u8>().dangling().as_ptr(),
        }
    }

    pub(crate) fn new(row_index: usize, ty: ComponentID, access: *mut u8) -> Self {
        Self {
            row_index,
            ty,
            access,
        }
    }

    pub(crate) fn cast<C: Component>(&self) -> &mut C {
        // assert_eq!(C::id(), self.ty);
        unsafe { self.access.cast::<C>().as_mut().unwrap() }
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
