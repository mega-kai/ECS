use crate::component::*;
use crate::storage::*;

#[derive(Debug, Clone, Copy)]
pub enum ExecutionFrequency {
    Always,
    Once,
    // Timed(f64, f64),
}

trait SystemParam {}
impl<'a> SystemParam for Command<'a> {}

struct Command<'a> {
    storage: &'a mut Storage,
}
impl<'a> Command<'a> {
    pub(crate) fn new() -> Self {
        todo!()
    }

    pub fn add_component<C: Component>(&mut self, component: C) {
        self.storage.add_component(component);
    }

    pub fn remove_and_discard_component<C: Component>(&mut self, key: ComponentKey) {
        let discard = self.storage.remove::<C>(key).unwrap();
    }
}

struct Query<'a> {
    storage: &'a mut Storage,
}
impl<'a> Query<'a> {}

trait System {
    type Input: SystemParam;
    fn run(&self, input: Self::Input);
}

pub struct Scheduler {
    updated: bool,
    pool: (),
    queue: (),
}
impl Scheduler {
    pub fn new() -> Self {
        Self {
            pool: (),
            queue: (),
            updated: false,
        }
    }

    pub fn add_system(&mut self) {
        self.updated = true;
    }

    pub fn generate_queue_for_current_tick(&mut self) {}

    pub fn execute_all(&mut self) {}
}
