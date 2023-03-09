use std::marker::PhantomData;
use std::slice;

use crate::component::*;
use crate::storage::*;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum ExecutionFrequency {
    Always,
    Once,
    // Timed(f64, f64),
}

pub struct Command<'a> {
    storage: &'a mut Storage,
}
impl<'a> Command<'a> {
    pub(crate) fn new(storage: &'a mut Storage) -> Self {
        Self { storage }
    }

    pub fn add_component<C: Component>(&mut self, component: C) {
        self.storage.add_component(component);
    }

    pub fn remove_component<C: Component>(&mut self, key: ComponentKey) -> C {
        self.storage.remove::<C>(key).unwrap()
    }

    pub fn query<Target: Component, Filter: Component>(&mut self) {}
}

// sortable with partial_eq
pub struct System {
    order: usize,
    frequency: ExecutionFrequency,
    func: fn(Command),
}
impl System {
    pub fn default(func: fn(Command)) -> Self {
        Self {
            order: 0,
            frequency: ExecutionFrequency::Always,
            func,
        }
    }

    pub fn new(order: usize, frequency: ExecutionFrequency, func: fn(Command)) -> Self {
        Self {
            order,
            frequency,
            func,
        }
    }

    pub(crate) fn run(&self, storage: &mut Storage) {
        (self.func)(Command::new(storage))
    }
}

// iterator and also indexible
struct QueryResult<C: Component> {
    phamtom: PhantomData<C>,
}

pub struct Scheduler {
    updated: bool,
    pool: Vec<System>,
    pub(crate) queue: Vec<System>,
}
impl Scheduler {
    pub fn new() -> Self {
        Self {
            pool: vec![],
            queue: vec![],
            updated: false,
        }
    }

    pub fn add_system(&mut self) {
        self.updated = true;
    }

    pub(crate) fn generate_queue(&mut self) {}
}
