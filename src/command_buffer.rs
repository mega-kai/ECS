//system level command buffer and ecs level command buffer
use crate::{component::*, system::*};

/// a command buffer for a single system, supplied to all systems upon
/// execution, returned by that function to be collected within
/// the main ECS::CommandBuffer;
/// all commands here will only get executed at a later stage of a tick
/// cycle, which is after the execution phase
pub struct Command {}
impl Command {
    //only available to other modules
    pub(crate) fn new() -> Self {
        todo!()
    }

    pub fn spawn_component<C: Component>(&mut self, component: C) -> ComponentKey {
        todo!()
    }

    //should you need an exclusive access to that component to destroy it??
    pub fn destroy_component<C: Component>(&mut self, mut_access: &mut C) {}

    /// perhaps the storage also stores linking information alongside with
    /// the actual component data
    pub fn link_component(&mut self, one: ComponentKey, another: ComponentKey) {}

    pub fn unlink_component(&mut self) {}
}
impl SysParam for Command {}

//main command buffer
pub struct CommandBuffer {}
impl CommandBuffer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn take() {}
}
