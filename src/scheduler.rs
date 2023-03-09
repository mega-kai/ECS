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

    pub fn query<T: Component, F: Filter>(&mut self) {}
}

pub trait Filter {}
impl<C: Component> Filter for And<C> {}
impl<C: Component> Filter for Or<C> {}
impl<C: Component> Filter for Not<C> {}

impl<F0: Filter> Filter for (F0,) {}
impl<F0: Filter, F1: Filter> Filter for (F0, F1) {}
impl<F0: Filter, F1: Filter, F2: Filter> Filter for (F0, F1, F2) {}
impl<F0: Filter, F1: Filter, F2: Filter, F3: Filter> Filter for (F0, F1, F2, F3) {}
impl<F0: Filter, F1: Filter, F2: Filter, F3: Filter, F4: Filter> Filter for (F0, F1, F2, F3, F4) {}
impl<F0: Filter, F1: Filter, F2: Filter, F3: Filter, F4: Filter, F5: Filter> Filter
    for (F0, F1, F2, F3, F4, F5)
{
}
impl<F0: Filter, F1: Filter, F2: Filter, F3: Filter, F4: Filter, F5: Filter, F6: Filter> Filter
    for (F0, F1, F2, F3, F4, F5, F6)
{
}
impl<
        F0: Filter,
        F1: Filter,
        F2: Filter,
        F3: Filter,
        F4: Filter,
        F5: Filter,
        F6: Filter,
        F7: Filter,
    > Filter for (F0, F1, F2, F3, F4, F5, F6, F7)
{
}

pub struct And<C: Component> {
    phantom: PhantomData<C>,
}
pub struct Or<C: Component> {
    phantom: PhantomData<C>,
}
pub struct Not<C: Component> {
    phantom: PhantomData<C>,
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

    pub fn add_system(&mut self, system: System) {
        self.pool.push(system);
        self.updated = true;
    }

    pub(crate) fn prepare_queue(&mut self) {
        if self.updated {
            self.updated = false;
        } else {
        }
    }
}
