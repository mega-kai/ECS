#![allow(dead_code, unused_variables)]
#![feature(core_intrinsics, portable_simd)]
use std::{any::TypeId, fmt::Debug};

//for serde
trait Component: Debug + Copy + Clone + 'static {}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
struct Key {
    index: usize,
    generation: usize,
}
impl Key {
    fn new(index: usize) -> Self {
        Self {
            index,
            generation: 0,
        }
    }
}

struct TypeErasedVec {}
impl TypeErasedVec {
    fn push() {}
    fn get() {}
    fn get_mut() {}
    fn remove() {}
}

struct SparseSet {
    dense: TypeErasedVec,
    //usize = TypeErasedVec.index,
    sparse: Vec<usize>,
    id: TypeId,
}
impl SparseSet {
    fn add() {}
    fn remove() {}
    fn get() {}
    fn get_mut() {}
}

struct Scheduler {
    pool: Vec<fn()>,
}
impl Scheduler {
    fn new() -> Self {
        Self { pool: vec![] }
    }

    fn add_system(&mut self, func: fn()) {
        self.pool.push(func);
    }

    fn run_all(&self) {
        for system in &self.pool {
            (system)()
        }
    }
}

struct ECS {
    storage: Vec<SparseSet>,
    scheduler: Scheduler,
}
impl ECS {
    fn new() -> Self {
        Self {
            storage: vec![],
            scheduler: Scheduler::new(),
        }
    }

    fn next(&mut self) {
        self.scheduler.run_all();
    }

    fn add_system(&mut self, func: fn()) {
        self.scheduler.add_system(func);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Test(i32);
    impl Component for Test {}

    fn test() {
        println!("hello ecs!")
    }

    #[test]
    fn sparse_set() {
        let mut ecs = ECS::new();
        ecs.add_system(test);
        ecs.next();
    }
}
