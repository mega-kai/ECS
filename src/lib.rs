#![allow(
    dead_code,
    unused_variables,
    unreachable_code,
    unused_mut,
    unused_assignments,
    unused_imports
)]
#![feature(alloc_layout_extra, core_intrinsics, portable_simd)]

// todo, thread safety, meaning all that access to anything within table shouldn't cause data race
// preferably using atomics???

use core::panic;
// use impls::impls;
use std::alloc::{alloc, dealloc, realloc};
use std::collections::HashMap;
use std::intrinsics::type_id;
use std::marker::PhantomData;
use std::mem::{forget, MaybeUninit};
use std::ops::{BitAnd, BitOr, BitXor, Deref, DerefMut, Not, Range};
use std::ptr::copy;
use std::simd::Simd;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::{alloc::Layout, fmt::Debug};

type SparseIndex = usize;
type DenseIndex = usize;
type BufferIndex = usize;
type TypeId = u128;
type Generation = usize;

const MASK_HEAD: usize = 1 << (usize::BITS - 1);
const MASK_TAIL: usize = !MASK_HEAD;

const SIMD_START: Simd<usize, 64> = Simd::from_array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]);

const SHIFT: Simd<usize, 64> = Simd::from_array([63; 64]);
const ONE: Simd<usize, 64> = Simd::from_array([1; 64]);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operation {
    And,
    Or,
    Xor,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Node {
    // index in ids
    Single(bool, u8),
    Dual(bool, u8, Operation, u8),
}

impl Debug for Filter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Filter")
            .field("num_of_nodes", &self.num_of_nodes)
            .field("num_of_ids", &self.num_of_ids)
            .field("nodes", &self.nodes)
            // .field("ids", &self.ids)
            .finish()
    }
}

const NUM_OF_NODES: usize = 16;
#[derive(Clone, PartialEq, Eq)]
pub struct Filter {
    num_of_nodes: u8,
    num_of_ids: u8,

    nodes: [Option<Node>; NUM_OF_NODES],
    ids: [TypeId; NUM_OF_NODES],
}
impl Filter {
    pub fn from<C: Clone + 'static>() -> Self {
        let mut thing = Self {
            num_of_nodes: 1,
            num_of_ids: 1,
            nodes: [None; NUM_OF_NODES],
            ids: [0; NUM_OF_NODES],
        };
        thing.ids[0] = std::intrinsics::type_id::<C>();
        // first node would be the root node
        thing.nodes[0] = Some(Node::Single(false, 0));
        thing
    }

    fn combine(mut first: Self, second: Self, op: Operation) -> Self {
        assert!(first.nodes[0].is_some() == true);
        assert!(second.nodes[0].is_some() == true);

        first.nodes.copy_within(0..first.num_of_nodes as usize, 1);
        first.nodes[0] = None;
        for each in &mut first.nodes[1..(1 + first.num_of_nodes) as usize] {
            match each {
                Some(val) => match val {
                    Node::Single(_, _) => (),
                    Node::Dual(_, index_left, _, index_right) => {
                        match index_left {
                            finally => *finally += 1,
                        }
                        match index_right {
                            finally => *finally += 1,
                        }
                    }
                },
                None => panic!("should not be none"),
            }
        }
        first.num_of_nodes += 1;

        first.nodes
            [first.num_of_nodes as usize..(first.num_of_nodes + second.num_of_nodes) as usize]
            .clone_from_slice(&second.nodes[0..second.num_of_nodes as usize]);
        for each in &mut first.nodes
            [first.num_of_nodes as usize..(first.num_of_nodes + second.num_of_nodes) as usize]
        {
            match each {
                Some(val) => match val {
                    Node::Single(_, _) => (),
                    Node::Dual(_, index_left, _, index_right) => {
                        match index_left {
                            finally => *finally += first.num_of_nodes,
                        }
                        match index_right {
                            finally => *finally += first.num_of_nodes,
                        }
                    }
                },
                None => panic!("should not be none"),
            }
        }
        first.nodes[0] = Some(Node::Dual(false, 1, op, first.num_of_nodes));
        first.num_of_nodes += second.num_of_nodes;

        first.ids
            [first.num_of_ids as usize..first.num_of_ids as usize + second.num_of_ids as usize]
            .clone_from_slice(&second.ids[0..second.num_of_ids as usize]);
        for each in &mut first.nodes
            [(first.num_of_nodes - second.num_of_nodes) as usize..first.num_of_nodes as usize]
        {
            match each {
                Some(val) => match val {
                    Node::Single(_, index) => *index += first.num_of_ids,
                    Node::Dual(_, _, _, _) => (),
                },
                None => panic!("should not be none"),
            }
        }
        first.num_of_ids += second.num_of_ids;

        first
    }

    const NULL: Filter = Self {
        num_of_nodes: 0,
        num_of_ids: 0,
        nodes: [None; NUM_OF_NODES],
        ids: [0; NUM_OF_NODES],
    };
}
impl BitAnd for Filter {
    type Output = Filter;
    fn bitand(self, rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::And)
    }
}
impl BitOr for Filter {
    type Output = Filter;
    fn bitor(self, rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::Or)
    }
}
impl BitXor for Filter {
    type Output = Filter;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::Xor)
    }
}
impl Not for Filter {
    type Output = Filter;
    fn not(mut self) -> Self::Output {
        match &mut self.nodes[0] {
            Some(val) => match val {
                Node::Single(flag, _) => *flag = !*flag,
                Node::Dual(flag, _, _, _) => *flag = !*flag,
            },
            None => panic!("invalid filter"),
        }
        self
    }
}

struct Column {
    ptr: *mut u8,
    cap: usize,

    gen_ptr: *mut u8,
}
impl Column {
    /// it is guranteed that the column would be zeroed and can be diveded by 64
    fn new(size: usize) -> Self {
        assert!(size % 64 == 0);
        let result = Self {
            ptr: unsafe { alloc(Layout::new::<usize>().repeat(size).unwrap().0) },
            gen_ptr: unsafe { alloc(Layout::new::<usize>().repeat(size).unwrap().0) },
            cap: size,
        };

        result.as_simd().fill(Simd::splat(0));
        result.as_gen_simd().fill(Simd::splat(0));
        result
    }

    fn double(&mut self) {
        unsafe {
            let ptr = realloc(
                self.ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
                Layout::new::<usize>()
                    .repeat(self.cap * 2)
                    .unwrap()
                    .0
                    .size(),
            );
            if ptr.is_null() {
                panic!("nullptr")
            }
            self.ptr = ptr;

            let gen_ptr = realloc(
                self.gen_ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
                Layout::new::<usize>()
                    .repeat(self.cap * 2)
                    .unwrap()
                    .0
                    .size(),
            );
            if gen_ptr.is_null() {
                panic!("nullptr")
            }
            self.gen_ptr = gen_ptr;
        };

        self.cap *= 2;
        self.as_simd()[self.cap / (64 * 2)..].fill(Simd::splat(0));
        self.as_gen_simd()[self.cap / (64 * 2)..].fill(Simd::splat(0));
    }

    fn as_slice(&self) -> &mut [usize] {
        unsafe { from_raw_parts_mut(self.ptr as *mut usize, self.cap) }
    }

    // this is also very very dangerous
    fn as_simd<'a, 'b>(&'a self) -> &'b mut [Simd<usize, 64>] {
        unsafe { from_raw_parts_mut(self.ptr as *mut Simd<usize, 64>, self.cap / 64) }
    }

    fn as_gen_slice(&self) -> &mut [Generation] {
        unsafe { from_raw_parts_mut(self.gen_ptr as *mut usize, self.cap) }
    }

    fn as_gen_simd<'a, 'b>(&'a self) -> &'b mut [Simd<Generation, 64>] {
        unsafe { from_raw_parts_mut(self.gen_ptr as *mut Simd<usize, 64>, self.cap / 64) }
    }
}
impl Drop for Column {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, Layout::new::<usize>().repeat(self.cap).unwrap().0) }
    }
}

struct ZST(u8);
const ZST_LAYOUT: Layout = Layout::new::<ZST>();

struct DenseColumn {
    ptr: *mut u8,
    sparse_ptr: *mut u8,
    len: usize,
    cap: usize,
    layout: Layout,
}
impl DenseColumn {
    fn new<C: 'static + Sized>() -> Self {
        let layout = Layout::new::<C>();
        if layout.size() > 0 {
            unsafe {
                Self {
                    ptr: alloc(layout.repeat(4).unwrap().0),
                    sparse_ptr: alloc(Layout::new::<usize>().repeat(4).unwrap().0),
                    len: 0,
                    cap: 4,
                    layout,
                }
            }
        } else {
            unsafe {
                Self {
                    ptr: alloc(ZST_LAYOUT.repeat(4).unwrap().0),
                    sparse_ptr: alloc(Layout::new::<usize>().repeat(4).unwrap().0),
                    len: 0,
                    cap: 4,
                    layout: ZST_LAYOUT,
                }
            }
        }
    }

    // maybe you can box leak this whole thing and reclaim it if doing some change, which also sounds like a massive hack
    fn double(&mut self) {
        unsafe {
            let ptr = realloc(
                self.ptr,
                self.layout.repeat(self.cap).unwrap().0,
                self.layout.repeat(self.cap * 2).unwrap().0.size(),
            );
            if ptr.is_null() {
                panic!("null ptr")
            }
            self.ptr = ptr;

            let ptr_sparse = realloc(
                self.sparse_ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
                Layout::new::<usize>()
                    .repeat(self.cap * 2)
                    .unwrap()
                    .0
                    .size(),
            );
            if ptr_sparse.is_null() {
                panic!("null ptr")
            }
            self.sparse_ptr = ptr_sparse;
        }
        self.cap *= 2;
    }

    /// returns a "naive" dense index, also super dangerous
    fn push<'a, 'b, C: 'static + Sized>(
        &'a mut self,
        value: C,
        sparse_index: SparseIndex,
    ) -> (DenseIndex, &'b mut C) {
        if self.len >= self.cap {
            self.double();
        }
        let dense_index = self.len;
        self.len += 1;
        self.as_sparse_slice()[dense_index] = sparse_index;
        unsafe {
            if self.layout == ZST_LAYOUT {
                copy(&ZST(0u8), &mut self.as_slice::<ZST>()[dense_index], 1);
            } else {
                copy(&value, &mut self.as_slice::<C>()[dense_index], 1);
                forget(value);
            }
        }
        (dense_index, &mut self.as_slice::<C>()[dense_index])
    }

    fn pop<C: 'static + Sized>(&mut self) -> C {
        unsafe {
            let mut value: MaybeUninit<C> = MaybeUninit::uninit();
            if self.layout != ZST_LAYOUT {
                copy::<C>(self.as_slice::<C>().last().unwrap(), value.as_mut_ptr(), 1);
            }
            self.len -= 1;
            value.assume_init()
        }
    }

    // don't think this is zst safe yet
    fn swap_remove<C: 'static + Sized>(&mut self, dense_index: DenseIndex) -> C {
        unsafe {
            let mut value: MaybeUninit<C> = MaybeUninit::uninit();
            if self.layout != ZST_LAYOUT {
                copy::<C>(
                    &mut self.as_slice::<C>()[dense_index],
                    value.as_mut_ptr(),
                    1,
                );
                copy::<C>(
                    self.as_slice::<C>().last().unwrap(),
                    &mut self.as_slice::<C>()[dense_index],
                    1,
                );
            }
            self.as_sparse_slice()[dense_index] = *self.as_sparse_slice().last().unwrap();
            self.len -= 1;
            value.assume_init()
        }
    }

    /// omg this is wildly unsafe, note that slicing into a zst would only yield a slice of 0u8s
    fn as_slice<'a, 'b, C: 'static + Sized>(&'a self) -> &'b mut [C] {
        let layout = Layout::new::<C>();
        if layout == self.layout || layout.size() == 0 {
            unsafe { from_raw_parts_mut(self.ptr as *mut C, self.len) }
        } else {
            panic!("type not matching")
        }
    }

    fn as_sparse_slice(&self) -> &mut [usize] {
        unsafe { from_raw_parts_mut(self.sparse_ptr as *mut usize, self.len) }
    }
}

impl Drop for DenseColumn {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout.repeat(self.cap).unwrap().0);
            dealloc(
                self.sparse_ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
            );
        }
    }
}

struct State {
    ptr: *mut u8,
    layout: Layout,
}
impl State {
    fn new<C: 'static + Sized>(val: C) -> Self {
        unsafe {
            let layout = Layout::new::<C>();
            let ptr = alloc(layout);
            copy::<C>(&val, ptr as *mut C, 1);
            forget(val);
            Self { ptr, layout }
        }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

pub struct Table {
    helpers: Vec<Column>,
    freehead: SparseIndex,
    table: HashMap<TypeId, (Column, DenseColumn)>,
    states: HashMap<TypeId, State>,
}

impl Table {
    fn new() -> Self {
        let mut result = Self {
            table: HashMap::new(),
            states: HashMap::new(),
            helpers: vec![],
            freehead: 1,
        };

        for _ in 0..10 {
            result.helpers.push(Column::new(64));
        }
        // 0 -> available, 1 -> in use
        // sparse column, num of components column, buffer availability column, 0s and 1s
        result.helpers[0].as_simd()[0] = SIMD_START;
        result.helpers[2].as_slice()[0..5].fill(1);
        result.helpers[4].as_simd().fill(Simd::splat(1));

        result
    }

    fn cap(&self) -> usize {
        self.helpers[0].cap
    }

    fn double(&mut self) {
        for each in &mut self.helpers {
            each.double();
        }
        for (_, (each, _)) in &mut self.table {
            each.double();
        }

        for each in self.cap() / (2 * 64)..self.cap() / 64 {
            self.helpers[0].as_simd()[each] = SIMD_START + Simd::splat(64 * each);
        }
    }

    fn apply_operation_then_cache(
        &mut self,
        left: BufferIndex,
        right: BufferIndex,
        invert_flag: bool,
        op: Operation,
    ) -> BufferIndex {
        let left_simd = self.helpers[left].as_simd();
        let right_simd = self.helpers[right].as_simd();
        let mut write_buffer_index = 0;
        for each in 5..self.helpers.len() {
            if self.helpers[2].as_slice()[each] == 0 {
                self.helpers[2].as_slice()[each] = 1;
                write_buffer_index = each;
                break;
            }
        }
        if write_buffer_index == 0 {
            self.helpers.push(Column::new(self.cap()));
            write_buffer_index = self.helpers.len() - 1;
        }
        let write_buffer_simd = self.helpers[write_buffer_index].as_simd();
        match op {
            Operation::And => {
                for each in 0..self.cap() / 64 {
                    write_buffer_simd[each] = left_simd[each] & right_simd[each]
                }
            }
            Operation::Or => {
                for each in 0..self.cap() / 64 {
                    write_buffer_simd[each] = left_simd[each] | right_simd[each]
                }
            }
            Operation::Xor => {
                for each in 0..self.cap() / 64 {
                    write_buffer_simd[each] = left_simd[each] ^ right_simd[each]
                }
            }
        }
        if invert_flag {
            for each in 0..self.cap() / 64 {
                write_buffer_simd[each] = !write_buffer_simd[each];
            }
        }
        // make sure these are not preset columns
        if left != 3 && left != 4 {
            self.helpers[2].as_slice()[left] = 0;
        }
        if right != 3 && right != 4 {
            self.helpers[2].as_slice()[right] = 0;
        }
        write_buffer_index
    }

    fn filter_traverse(&mut self, node_index: usize, filter: &Filter) -> BufferIndex {
        match &filter.nodes[node_index].unwrap() {
            Node::Single(invert_flag, id_index) => {
                match self.table.get(&filter.ids[*id_index as usize]) {
                    Some((sparse_column, _)) => {
                        let mut buffer_index = 0;
                        for each in 5..self.helpers.len() {
                            if self.helpers[2].as_slice()[each] == 0 {
                                self.helpers[2].as_slice()[each] = 1;
                                buffer_index = each;
                                break;
                            }
                        }
                        if buffer_index == 0 {
                            self.helpers.push(Column::new(self.cap()));
                            buffer_index = self.helpers.len() - 1;
                        }
                        let buffer_simd = self.helpers[buffer_index].as_simd();
                        buffer_simd.clone_from_slice(&sparse_column.as_simd());
                        if *invert_flag {
                            for each in 0..self.cap() / 64 {
                                buffer_simd[each] = !buffer_simd[each];
                            }
                        }
                        buffer_index
                    }
                    None => {
                        if *invert_flag {
                            4
                        } else {
                            3
                        }
                    }
                }
            }

            Node::Dual(invert_flag, node_index_left, op, node_index_right) => {
                let left = self.filter_traverse(*node_index_left as usize, filter);
                let right = self.filter_traverse(*node_index_right as usize, filter);
                // each time you call this only one of the buffers will be freed, after all the branches are
                // resolved only one buffer will remain
                self.apply_operation_then_cache(left, right, *invert_flag, *op)
            }
        }
    }

    fn free_all_buffers(&mut self) {
        for each in 5..self.helpers.len() {
            self.helpers[2].as_slice()[each] = 0;
        }
    }

    /// if u want to init a column with nothing in it
    pub fn register_column<C: 'static + Sized>(&mut self) {
        let cap = self.cap();
        self.table
            .entry(type_id::<C>())
            .or_insert((Column::new(cap), DenseColumn::new::<C>()));
    }
    /// the lifetime isn't locked
    pub unsafe fn read_column_raw<'a, 'b, C: 'static + Sized>(
        &'a self,
    ) -> Result<&'b [C], &'static str> {
        match self.table.get(&type_id::<C>()) {
            Some((_, dense_column)) => Ok(dense_column.as_slice()),
            None => Err("type not in table"),
        }
    }

    pub fn insert_new<C: 'static + Sized>(&mut self, value: C) -> Access<C> {
        let sparse_index = self.freehead;
        for each in sparse_index + 1..self.cap() {
            if self.helpers[1].as_slice()[each] == 0 {
                self.freehead = each;
                break;
            }
        }
        if sparse_index == self.freehead {
            self.double();
            self.freehead = self.cap() / 2;
        }
        let cap = self.cap();

        let (target_sparse, targe_dense) = self
            .table
            .entry(type_id::<C>())
            .or_insert((Column::new(cap), DenseColumn::new::<C>()));

        let (dense_index, _) = targe_dense.push(value, sparse_index);
        target_sparse.as_slice()[sparse_index] = dense_index | MASK_HEAD;
        // update generation
        target_sparse.as_gen_slice()[sparse_index] += 1;
        let gen = target_sparse.as_gen_slice()[sparse_index];
        targe_dense.as_sparse_slice()[dense_index] = sparse_index;
        self.helpers[1].as_slice()[sparse_index] += 1;
        Access::new(self, AccessType::Cell(sparse_index), gen)
    }
    pub fn insert_at<C: 'static + Sized>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<Access<C>, &'static str> {
        let cap = self.cap();
        let (target_sparse, target_dense) = self
            .table
            .entry(type_id::<C>())
            .or_insert((Column::new(cap), DenseColumn::new::<C>()));
        match target_sparse.as_slice()[sparse_index] {
            0 => {
                let (dense_index, _) = target_dense.push(value, sparse_index);
                target_sparse.as_slice()[sparse_index] = dense_index | MASK_HEAD;
                target_sparse.as_gen_slice()[sparse_index] += 1;
                let gen = target_sparse.as_gen_slice()[sparse_index];
                target_dense.as_sparse_slice()[dense_index] = sparse_index;
                self.helpers[1].as_slice()[sparse_index] += 1;
                Ok(Access::new(self, AccessType::Cell(sparse_index), gen))
            }
            _ => Err("sparse index at this type column is already taken"),
        }
    }
    pub fn remove<C: 'static + Sized>(&mut self, access: Access<C>) -> Result<C, &'static str> {
        let sparse_index = access
            .get_sparse_index()
            .ok_or("this is an access for state not for cells")?;
        let (target_sparse, target_dense) = self
            .table
            .get_mut(&type_id::<C>())
            .ok_or("type not in table")?;
        if sparse_index == 0 || sparse_index >= target_sparse.cap {
            return Err("invalid sparse index");
        }
        if target_sparse.as_gen_slice()[sparse_index] != access.gen {
            return Err("invalid generation");
        }
        match target_sparse.as_slice()[sparse_index] {
            0 => Err("sparse index at this type column is empty"),
            _ => {
                let dense_index = target_sparse.as_slice()[sparse_index] & MASK_TAIL;
                target_sparse.as_slice()[sparse_index] = 0;
                self.helpers[1].as_slice()[sparse_index] -= 1;
                if self.helpers[1].as_slice()[sparse_index] == 0 && sparse_index < self.freehead {
                    self.freehead = sparse_index;
                }
                // last element
                let value = if dense_index == target_dense.len - 1 {
                    target_dense.pop()
                } else {
                    let val = target_dense.swap_remove::<C>(dense_index);
                    let sparse_index_to_change = target_dense.as_sparse_slice()[dense_index];
                    target_sparse.as_slice()[sparse_index_to_change] = dense_index | MASK_HEAD;
                    val
                };
                Ok(value)
            }
        }
    }

    // states don't have generations; this is based on hashmap so it actually supports zst states
    pub fn add_state<C: 'static + Sized>(&mut self, res: C) -> Result<Access<C>, &'static str> {
        if let Some(_) = self.states.get(&type_id::<C>()) {
            Err("state already present in table")
        } else {
            self.states.insert(type_id::<C>(), State::new::<C>(res));
            Ok(Access::new(self, AccessType::State, !0))
        }
    }
    pub fn read_state<C: 'static + Sized>(&mut self) -> Result<Access<C>, &'static str> {
        if let Some(_) = self.states.get(&type_id::<C>()) {
            Ok(Access::new(self, AccessType::State, !0))
        } else {
            Err("state never registered")
        }
    }
    pub fn remove_state<C: 'static + Sized>(&mut self) -> Result<C, &'static str> {
        if let Some(state) = self.states.remove(&type_id::<C>()) {
            unsafe {
                let mut value: MaybeUninit<C> = MaybeUninit::uninit();
                copy::<C>(
                    (state.ptr as *mut C).as_mut().unwrap(),
                    value.as_mut_ptr(),
                    1,
                );
                Ok(value.assume_init())
            }
        } else {
            Err("state not registered")
        }
    }

    unsafe fn read_raw<C: 'static + Sized>(
        &self,
        sparse_index: SparseIndex,
        gen: Generation,
    ) -> &mut C {
        let (sparse_column, dense_column) = self.table.get(&type_id::<C>()).unwrap();
        if sparse_column.as_gen_slice()[sparse_index] != gen {
            panic!("generation not matching")
        }
        &mut dense_column.as_slice::<C>()[sparse_column.as_slice()[sparse_index] & MASK_TAIL]
    }
    unsafe fn read_state_raw<C: 'static + Sized>(&self) -> &mut C {
        (self.states.get(&type_id::<C>()).unwrap().ptr as *mut C)
            .cast::<C>()
            .as_mut()
            .unwrap()
    }

    // todo intermediate product, can yield IterMut or can be further queried by each method
    pub fn query_with_filter(&mut self, filter: &Filter) -> IterMut {
        let result_index = self.filter_traverse(0, filter);
        let result_buffer_simd = self.helpers[result_index].as_simd();
        let sparse_simd = self.helpers[0].as_simd();
        for each in 0..self.cap() / 64 {
            result_buffer_simd[each] =
                !((result_buffer_simd[each] >> SHIFT) - ONE) & sparse_simd[each];
        }
        let result_buffer_slice = self.helpers[result_index].as_slice(); // [..largest_occupied_sparse_index]
        result_buffer_slice.sort_unstable_by(|a, b| b.cmp(a));
        IterMut::new(result_buffer_slice.as_ptr_range())
    }

    // which is unfortunately, impossible to implement, unless you can check trait for typeid
    pub fn query_with_trait() {
        todo!()
    }

    // todo, load/save a single column, then reassemble into a whole table.
    // what if we use a savable trait that has a method to save/load stuff
    pub fn load_column<C: 'static + Sized>() {
        todo!()
    }
    pub fn save_column<C: 'static + Sized>() {
        todo!()
    }
}

enum AccessType {
    State,
    Cell(SparseIndex),
}
// why did i use access in the first place????
pub struct Access<C: 'static + Sized> {
    _phantom: PhantomData<C>,
    table: *const Table,
    ty: AccessType,
    gen: Generation,
}
impl<C: 'static + Sized> Access<C> {
    fn new(table: *const Table, ty: AccessType, gen: Generation) -> Self {
        Self {
            _phantom: PhantomData,
            table,
            ty,
            gen,
        }
    }

    pub fn get_sparse_index(&self) -> Option<SparseIndex> {
        match self.ty {
            AccessType::Cell(index) => Some(index),
            AccessType::State => None,
        }
    }
}
impl<C: 'static + Sized> Deref for Access<C> {
    type Target = C;
    fn deref(&self) -> &Self::Target {
        match self.ty {
            AccessType::Cell(sparse_index) => unsafe {
                (*self.table).read_raw::<C>(sparse_index, self.gen)
            },
            AccessType::State => unsafe { (*self.table).read_state_raw::<C>() },
        }
    }
}
impl<C: 'static + Sized> DerefMut for Access<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.ty {
            AccessType::Cell(sparse_index) => unsafe {
                (*self.table).read_raw::<C>(sparse_index, self.gen)
            },
            AccessType::State => unsafe { (*self.table).read_state_raw::<C>() },
        }
    }
}
impl<C: 'static + Sized> AsRef<C> for Access<C> {
    fn as_ref(&self) -> &C {
        &*self
    }
}
impl<C: 'static + Sized> AsMut<C> for Access<C> {
    fn as_mut(&mut self) -> &mut C {
        &mut *self
    }
}

pub struct IterMut {
    ptr: *const usize,
    end: *const usize,
}
impl Debug for IterMut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slice = unsafe { from_raw_parts(self.ptr, self.end.offset_from(self.ptr) as usize) };
        f.debug_struct("IterMutAlt")
            .field("slice: ", &slice)
            .finish()
    }
}
impl IterMut {
    fn new(range: Range<*const usize>) -> Self {
        Self {
            ptr: range.start,
            end: range.end,
        }
    }
}
impl Iterator for IterMut {
    type Item = SparseIndex;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ptr != self.end && *self.ptr != 0 {
                let index = *self.ptr;
                self.ptr = self.ptr.add(1);
                Some(index)
            } else {
                None
            }
        }
    }
}

//-----------------ECS-----------------//
pub struct ECS {
    pub table: Table,
    entry_point: fn(&mut Table),
}

impl ECS {
    pub fn new(entry_point: fn(&mut Table)) -> Self {
        Self {
            table: Table::new(),
            entry_point,
        }
    }

    pub fn tick(&mut self) {
        (self.entry_point)(&mut self.table);
        self.table.free_all_buffers();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // use impls::impls;
    trait UwU {}
    impl UwU for u32 {}

    #[test]
    fn test() {
        // assert!(impls!(u32: Copy & Debug & UwU));
    }
}
