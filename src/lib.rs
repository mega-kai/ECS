#![allow(
    dead_code,
    unused_variables,
    unreachable_code,
    unused_mut,
    unused_assignments
)]
#![feature(
    alloc_layout_extra,
    allocator_api,
    map_try_insert,
    core_intrinsics,
    const_trait_impl,
    const_type_id,
    const_mut_refs,
    const_type_name,
    portable_simd,
    array_zip,
    slice_as_chunks
)]

use core::panic;
use hashbrown::HashMap;
use std::alloc::{alloc, dealloc, realloc};
use std::intrinsics::type_id;
use std::mem::size_of;
use std::ops::{BitAnd, BitOr, BitXor, Not, Range};
use std::simd::Simd;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::{alloc::Layout, fmt::Debug};
// caching idea, you can treat a row cache as a bitset indexed by a type id;
// you can run that row cache thru the filter and yield a boolean value, which would determine if that row fits that filter;
// this way you can cache all filters within the table and also cache all the rows and update them and run the updated row
// thru the cached filters to see if they fit (remember to eliminate repeated filters by only hashing the computed id)
// but a problem is that you need to loop thru all the cached filters all the time, which is very expensive

// what is this paged array thing? i suspect that the os is having trouble issuing a continuous memory block for 30000 i32s

// you can also circumvent the possible query overhead with multiple tables in the same system; tho these are more like worlds than scenes

// for a closed world design consisted of numerous scenes you'd have a current loaded scene where each entity must be loaded in full
// granularity while the background scenes can adapt a level of detail approximated update scheme
// for an open world design you'd have a loaded area and rest of the world being dormant, loaded area obviously would need a full
// granularity in the ecs but the rest can also just adapt a LoD pattern
// how should this approximation algorithm be implemented? one obvious requires a load screen and the other has a dynamic loading pattern
// and it seems the current insertion/removal design suits better with the open world one, maybe a "swap scene" and load the next scene

//-----------------STORAGE-----------------//
type SparseIndex = usize;
type DenseIndex = usize;
type DenseColumnIndex = usize;
type BufferIndex = usize;
type TypeId = u64;

const MASK_HEAD: usize = 1 << (usize::BITS - 1);
const MASK_TAIL: usize = !MASK_HEAD;

const SIMD_START: Simd<usize, 64> = Simd::from_array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]);

const SHIFT: Simd<usize, 64> = Simd::from_array([63; 64]);
const ONE: Simd<usize, 64> = Simd::from_array([1; 64]);

//-----------------FILTER-----------------//
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
    pub const fn from<C: Clone + 'static>() -> Self {
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
}
impl Column {
    /// it is guranteed that the column would be zeroed and can be diveded whole by 64
    fn new(size: usize) -> Self {
        assert!(size % 64 == 0);
        let mut result = Self {
            ptr: unsafe { alloc(Layout::new::<usize>().repeat(size).unwrap().0) },
            cap: size,
        };
        result.as_simd().fill(Simd::splat(0));
        result
    }

    fn double(&mut self) {
        self.ptr = unsafe {
            realloc(
                self.ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
                size_of::<usize>() * self.cap * 2,
            )
        };
        self.cap *= 2;
        self.as_simd()[self.cap / (64 * 2)..].fill(Simd::splat(0));
    }

    fn as_slice<'a, 'b>(&'a self) -> &'b mut [usize] {
        unsafe { from_raw_parts_mut(self.ptr as *mut usize, self.cap) }
    }

    fn as_simd<'a, 'b>(&'a self) -> &'b mut [Simd<usize, 64>] {
        unsafe { from_raw_parts_mut(self.ptr as *mut Simd<usize, 64>, self.cap / 64) }
    }
}
impl Drop for Column {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, Layout::new::<usize>().repeat(self.cap).unwrap().0) }
    }
}

struct DenseColumn {
    ptr: *mut u8,
    sparse_ptr: *mut u8,
    len: usize,
    cap: usize,
    layout: Layout,
}
impl DenseColumn {
    fn new<C: 'static + Clone + Sized>() -> Self {
        let layout = Layout::new::<C>();
        unsafe {
            Self {
                ptr: alloc(layout.repeat(8).unwrap().0),
                sparse_ptr: alloc(Layout::new::<usize>().repeat(8).unwrap().0),
                len: 0,
                cap: 8,
                layout,
            }
        }
    }

    fn double(&mut self) {
        unsafe {
            self.ptr = realloc(
                self.ptr,
                self.layout.repeat(self.cap).unwrap().0,
                self.layout.size() * self.cap * 2,
            );
            self.sparse_ptr = realloc(
                self.sparse_ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
                size_of::<usize>() * self.cap * 2,
            )
        }
        self.cap *= 2;
    }

    /// returns a "naive" dense index
    fn push<C: 'static + Clone + Sized>(
        &mut self,
        value: C,
        sparse_index: SparseIndex,
    ) -> DenseIndex {
        if self.len == self.cap {
            self.double();
        }
        let dense_index = self.len;
        self.len += 1;
        self.as_slice::<C>()[dense_index] = value;
        self.as_sparse_slice()[dense_index] = sparse_index;
        dense_index
    }

    /// pop both
    fn pop<C: 'static + Clone + Sized>(&mut self) -> C {
        let value = self.as_slice::<C>().last().unwrap().clone();
        self.len -= 1;
        value
    }

    fn swap_remove<C: 'static + Clone + Sized>(&mut self, dense_index: DenseIndex) -> C {
        let value = self.as_slice::<C>()[dense_index].clone();
        self.as_slice::<C>()[dense_index] = self.as_slice::<C>().last().unwrap().clone();
        self.as_sparse_slice()[dense_index] = self.as_sparse_slice().last().unwrap().clone();
        self.len -= 1;
        value
    }

    fn as_slice<'a, 'b, C: 'static + Clone + Sized>(&'a self) -> &'b mut [C] {
        assert!(Layout::new::<C>() == self.layout);
        unsafe { from_raw_parts_mut(self.ptr as *mut C, self.len) }
    }

    fn as_sparse_slice<'a, 'b>(&'a self) -> &'b mut [usize] {
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

// event is a temporary data that carries information across systems, an event is triggered when component/resources is changed
struct EventColumn {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    layout: Layout,
    seal: usize,
}
impl EventColumn {
    fn new_empty<C: 'static + Clone + Sized>() -> Self {
        unsafe {
            let layout = Layout::new::<C>();
            Self {
                ptr: alloc(layout.repeat(8).unwrap().0),
                len: 0,
                cap: 8,
                layout,
                seal: 0,
            }
        }
    }

    fn double(&mut self) {
        unsafe {
            self.ptr = realloc(
                self.ptr,
                self.layout.repeat(self.cap).unwrap().0,
                self.cap * 2 * self.layout.size(),
            );
            self.cap *= 2;
        }
    }

    /// this will reset the seal
    fn clear_sealed_then_reseal(&mut self) {
        let len = self.len - self.seal;
        unsafe {
            std::ptr::copy(
                self.ptr.add(self.seal * self.layout.size()),
                self.ptr,
                len * self.layout.size(),
            )
        };
        self.len = len;
        self.seal = self.len;
    }

    fn push<C: 'static + Clone + Sized>(&mut self, value: C) {
        if self.len == self.cap {
            self.double();
        }
        unsafe { *(self.ptr as *mut C).add(self.len) = value };
        self.len += 1;
    }

    fn pop<C: 'static + Clone + Sized>(&mut self) -> C {
        self.len -= 1;
        let value = unsafe { (self.ptr as *mut C).add(self.len).as_ref().unwrap().clone() };
        value
    }

    /// not mutable tho
    fn as_slice<'a, 'b, C: 'static + Clone + Sized>(&'a self) -> &'b [C] {
        unsafe { from_raw_parts(self.ptr as *mut C, self.len) }
    }
}
impl Drop for EventColumn {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout.repeat(self.cap).unwrap().0) }
    }
}

struct Resource {
    ptr: *mut u8,
    layout: Layout,
}
impl Resource {
    fn new<C: 'static + Clone + Sized>(val: C) -> Self {
        unsafe {
            let layout = Layout::new::<C>();
            let ptr = alloc(layout);
            *(ptr as *mut C) = val;
            Self { ptr, layout }
        }
    }
}
impl Drop for Resource {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

pub struct Table {
    helpers: Vec<Column>,
    freehead: SparseIndex,
    // largest_occupied_index: SparseIndex,
    // so when you can clip at the largest taken
    // also maybe regularly dense up the columns
    table: HashMap<TypeId, (Column, DenseColumn)>,
    resources: HashMap<TypeId, Resource>,
    events: HashMap<TypeId, EventColumn>,
}

impl Table {
    fn new() -> Self {
        let mut result = Self {
            table: HashMap::new(),
            resources: HashMap::new(),
            events: HashMap::new(),
            helpers: vec![],
            freehead: 1,
        };

        for each in 0..10 {
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

    pub fn insert_new<C: 'static + Clone + Sized>(&mut self, value: C) -> SparseIndex {
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
        let dense_index = targe_dense.push(value, sparse_index);
        target_sparse.as_slice()[sparse_index] = dense_index | MASK_HEAD;
        targe_dense.as_sparse_slice()[dense_index] = sparse_index;
        self.helpers[1].as_slice()[sparse_index] += 1;
        sparse_index
    }

    pub fn insert_at<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<(), &'static str> {
        let cap = self.cap();
        let (target_sparse, target_dense) = self
            .table
            .entry(type_id::<C>())
            .or_insert((Column::new(cap), DenseColumn::new::<C>()));
        match target_sparse.as_slice()[sparse_index] {
            0 => {
                let dense_index = target_dense.push(value, sparse_index);
                target_sparse.as_slice()[sparse_index] = dense_index | MASK_HEAD;
                target_dense.as_sparse_slice()[dense_index] = sparse_index;
                self.helpers[1].as_slice()[sparse_index] += 1;
                Ok(())
            }
            _ => Err("sparse index at this type column is already taken"),
        }
    }

    pub fn remove<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
    ) -> Result<C, &'static str> {
        let (target_sparse, target_dense) = self
            .table
            .get_mut(&type_id::<C>())
            .ok_or("type not in table")?;
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

    fn reseal_all(&mut self) {
        for (_, each) in &mut self.events {
            each.clear_sealed_then_reseal();
        }
    }

    // todo deal with null filters
    pub fn query(&mut self, filter: &Filter) -> IterMut {
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

    pub fn read<'a, 'b, C: 'static + Clone + Sized>(
        &'a self,
        sparse_index: SparseIndex,
    ) -> Option<&'b mut C> {
        match self.table.get(&type_id::<C>()) {
            Some((sparse_column, dense_column)) => Some(
                &mut dense_column.as_slice::<C>()
                    [sparse_column.as_slice()[sparse_index] & MASK_TAIL],
            ),
            None => None,
        }
    }

    pub fn add_resource<C: 'static + Clone + Sized>(&mut self, res: C) -> Result<(), &'static str> {
        if let Some(_) = self.resources.get(&type_id::<C>()) {
            Err("resource already present in table")
        } else {
            self.resources
                .insert(type_id::<C>(), Resource::new::<C>(res));
            Ok(())
        }
    }
    pub fn read_resource<'a, 'b, C: 'static + Clone + Sized>(
        &'a mut self,
    ) -> Result<&'b mut C, &'static str> {
        if let Some(res) = self.resources.get(&type_id::<C>()) {
            Ok(unsafe { (res.ptr as *mut C).cast::<C>().as_mut().unwrap() })
        } else {
            Err("resource not in table")
        }
    }
    pub fn remove_resource<C: 'static + Clone + Sized>(&mut self) -> Result<C, &'static str> {
        if let Some(res) = self.resources.remove(&type_id::<C>()) {
            Ok(unsafe { (res.ptr as *mut C).as_mut().unwrap().clone() })
        } else {
            Err("resource not in table")
        }
    }

    /// fire an event, will initiate a event stack if it's not already present in the table, then push the event to that stack
    pub fn add_event<C: 'static + Clone + Sized>(&mut self, event: C) {
        self.events
            .entry(type_id::<C>())
            .or_insert(EventColumn::new_empty::<C>())
            .push::<C>(event);
    }
    /// todo make this iterator queue compatible
    pub fn read_event<C: 'static + Clone + Sized>(&mut self) -> Result<IterMut, &'static str> {
        match self.events.get(&type_id::<C>()) {
            Some(queue) => Ok(IterMut::new(queue.as_slice().as_ptr_range())),
            None => Err("event type not in the column"),
        }
    }
    pub fn remove_event<C: 'static + Clone + Sized>(&mut self) -> Result<(), &'static str> {
        self.events
            .remove(&type_id::<C>())
            .ok_or("event type not in table")?;
        Ok(())
    }

    pub fn load() {
        todo!()
    }
    pub fn save() {
        todo!()
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
    // todo make it so the first time you query a filter it actually registers it,
    // and later ticks it would just get the cached filter
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
        self.table.reseal_all();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone, Debug)]
    struct Health(i32);
    const HEALTH: Filter = Filter::from::<Health>();

    #[derive(Clone, Debug, PartialEq)]
    struct Mana(i32);
    const MANA: Filter = Filter::from::<Mana>();

    #[derive(Clone, Debug)]
    struct Player(&'static str);
    const PLAYER: Filter = Filter::from::<Player>();

    #[derive(Clone, Debug)]
    struct Enemy(&'static str);
    const ENEMY: Filter = Filter::from::<Enemy>();
    // mutable access detection -> change detection????

    fn entry_point(table: &mut Table) {}
    #[test]
    fn ecs() {
        let mut ecs = ECS::new(entry_point);
    }
}
