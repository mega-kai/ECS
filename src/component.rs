use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    fmt::Debug,
    mem::size_of,
};

/// FOR QUERYING
///  return an iterator of &C/&mut C, querying a
/// single component by an id or index is not allowed;
///
/// should also be able to query for (&mut A, ..., With/Or/Without<&B>, ...)
/// which are basically archetypes;
///
/// a distinction between (&mut A, &B) and (&mut A, With<B>) is that
/// the latter does not issue a shared reference iterator of B, thus in
/// the same cycle an iterator of mutable refs can still be issued, as
/// With only guarantees that such component exists along with the
/// ones you are querying, while &B can give you read access to its data

/// a key to access
pub struct ComponentKey {
    index: usize,
    generation: usize,
    ty: ComponentID,
}

/// ID used for comparing component types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentID {
    name: &'static str,
    id: TypeId,
}

/// marker trait for components
/// TODO! proc macro derive
pub trait Component: Debug + Copy + Clone + 'static {
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

/// an iterator of results prepared by the
#[derive(Debug, Clone, Copy)]
pub struct QueryRequest {}
impl QueryRequest {
    pub fn empty() -> Self {
        Self {}
    }
}

/// impl intoiterator on this, iter() returns an iter of &Component
/// while iter_mut returns an iter of &mut Component
pub struct QueryResult(/* a bunch of references to components */);
impl QueryResult {
    pub fn new() -> Self {
        todo!()
    }

    pub fn does_things_with_the_result(&mut self) {}
}

/// queue up commands, those commands would be applied after all systems are executed
pub struct CommandQueue {}
impl CommandQueue {
    pub fn new() -> Self {
        todo!()
    }

    pub fn spawn_component(&mut self) {}

    pub fn destroy_component(&mut self) {}

    pub fn attach_component_to_another(&mut self) {}

    pub fn remove_component_from_another(&mut self) {}
}
