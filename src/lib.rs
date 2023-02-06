#![allow(dead_code, unused_variables)]
#![feature(core_intrinsics, portable_simd, alloc_layout_extra)]
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    collections::HashMap,
    fmt::Debug,
    num::NonZeroUsize,
    ptr::NonNull,
    vec::IntoIter,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ComponentID {
    name: &'static str,
    id: TypeId,
}
impl ComponentID {
    fn new<C: Component>() -> Self {
        Self {
            name: type_name::<C>(),
            id: TypeId::of::<C>(),
        }
    }
}

//for serde
trait Component: Debug + Copy + Clone + 'static {
    fn generate_id() -> ComponentID {
        ComponentID::new::<Self>()
    }
}

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

struct TypeErasedVec {
    //of the component type
    data_layout: Layout,
    //ptr to a byte
    data_heap_ptr: NonNull<u8>,
    //[1,inf]
    len: usize,
    capacity: usize,
}
impl TypeErasedVec {
    /// does not handle ZST
    fn new<T>() -> Self {
        assert!(std::mem::size_of::<T>() != 0, "it's a ZST");
        let layout = Layout::new::<T>();
        let mut val = Self {
            data_layout: layout,
            data_heap_ptr: layout.dangling(),
            len: 0,
            capacity: 0,
        };
        //all new vec gets 64 free space
        val.grow_capacity(unsafe { NonZeroUsize::new_unchecked(64) });
        val
    }

    fn push(&mut self, ptr: NonNull<u8>, layout: Layout) {
        assert_eq!(self.data_layout, layout);

        if self.len >= self.capacity {
            //double the cap
            self.grow_capacity(unsafe { NonZeroUsize::new_unchecked(self.capacity) });
            self.insert_from_ptr(self.len - 1, ptr);
        } else {
            self.insert_from_ptr(self.len - 1, ptr);
        }
    }

    /// try to make sure this is only used to double the capacity
    fn grow_capacity(&mut self, grow: NonZeroUsize) {
        let new_capacity = self.capacity + grow.get();
        let (new_layout, _) = self
            .data_layout
            .repeat(new_capacity)
            .expect("could not repeat this layout");
        let new_data_ptr = if self.capacity == 0 {
            unsafe { std::alloc::alloc(new_layout) }
        } else {
            unsafe {
                std::alloc::realloc(
                    //starting at
                    self.data_heap_ptr.as_ptr(),
                    //the extent to uproot
                    self.data_layout
                        .repeat(self.capacity)
                        .expect("could not repeat layout")
                        .0,
                    //length of the new memory
                    new_layout.size(),
                )
            }
        };
        self.capacity = new_capacity;
        self.data_heap_ptr = unsafe { NonNull::new_unchecked(new_data_ptr) };
    }

    fn insert_from_ptr(&mut self, index: usize, ptr: NonNull<u8>) {
        self.len += 1;
    }

    fn get(&self) {}
    fn get_mut(&mut self) {}
    fn remove(&mut self, index: usize) {}
    fn swap(&mut self) {}
}

struct SparseSet {
    dense: TypeErasedVec,
    //usize == TypeErasedVec.index,
    sparse: Vec<usize>,
}
impl SparseSet {
    fn add(&mut self, ptr: NonNull<u8>) {}
    fn remove() {}
    fn get() {}
    fn get_mut() {}
}
impl IntoIterator for SparseSet {
    type Item = u8;

    type IntoIter = IntoIter<u8>;

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

struct Storage {
    // no repeating items
    data_hash: HashMap<ComponentID, SparseSet>,
}
impl Storage {
    fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
        }
    }

    fn add_component<C: Component>(&mut self, key: Key, component: C) {
        let check_against = ComponentID::new::<C>();
        let result = self.data_hash.get_mut(&check_against);
        if let Some(access) = result {
            //add component
        } else {
            //create a new
        }
    }

    /// this function is supposed to return an iterator of either &C, &mut C or C
    fn query<C: Component>(&mut self) -> Option<()> {
        todo!()
    }
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
    storage: Storage,
    scheduler: Scheduler,
}
impl ECS {
    fn new() -> Self {
        Self {
            storage: Storage::new(),
            scheduler: Scheduler::new(),
        }
    }

    fn next(&mut self) {
        self.scheduler.run_all();
    }

    fn add_system(&mut self, func: fn()) {
        self.scheduler.add_system(func);
    }

    fn spawn<C: Component>(&mut self, component: C) -> Key {
        //generate key
        let key = Key::new(0);
        self.storage.add_component(key, component);
        key
    }

    fn query<C: Component>(&mut self) {
        self.storage.query::<C>();
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

        ecs.spawn(Test(12));
    }

    #[test]
    fn iter() {
        let mut iter = [12; 3].iter();
        iter.next();
    }
}
