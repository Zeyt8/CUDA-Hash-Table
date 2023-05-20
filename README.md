# CUDA Hash Table

## Implementation

I made a structure that holds the pair key-value. The hash table is an array of such pairs.

To treat collisions, I used linear probing. A pair is considered empty if the key is 0. That is because the hash map is made for keys and values that are greater than 0.

For each function the idea is to call a kernel that will process one of the elements.

For every function I have to move the array arguments to the GPU.

### **Insert**

If presumably all the keys are not present in the table and after adding them the count exceeds the size of the table, I have to resize the table.

I start a kernel. Each instance has an index. I calculate the hash of the key at that index. I do linear probing to find a spot. That means checking if the current hash is empty or if it has the same key. If it is empty or has the same key, I insert the new key-value pair. Otherwise I increase the hash by 1.

To check if the hash is empty I use the atomicCAS function. This prevents race conditions. The idea is that if the key is 0, I should swap it with the given key. That way I reserve that space and it is done atomically. Then by checking the previous value I can use ifs to determine further behaviour. Because I reserved the space, now I can do anything with it without needing further synchronization.

I keep track of how many new pairs I added.

In the end I update the new count.

### **Get**

I create a new array as managed, because I need it on the device, but I also need to return it on the host.

I start a kernel. Each instance has an index. It computes the hash of the key at that index and checks for existing values also using linear probing.

If it finds something, it puts the result in the result array passed to the function. Otherwise it puts 0.

### **Reshape**

I declare a new hash table. I start a kernel. Each instance has an index and checks if the old table had something at that index.

If it did, it calculates the new hash and inserts it in the new table. The insertion is done with linear probing aswell.

In the end I swap the 2 tables.

## GPU RAM

The hash table is only stored on the GPU using malloc. I do not do any operations on it on the host, except swapping it with the new one when reshaping, but that is only swapping pointers, not using it.

## Performance

It is expected for it to be much faster than the CPU. Surprinzingly using atomicCSA doesn't seem to slow it down that much. I think that happens because the operation does something hardware wise. If I were to use a classic mutex, it would be much slower, as most of the inserting and getting would be under that mutex so most of it wouldn't actually be multithreaded.

Because I am inserting a lot of data at a time, it is possible that reshaping that isn't needed often happens. I don't think this is fixable, as reshaping needs to happen before inserting the new data. Checking if it is needed would mean basically inserting the data twice, once to check if it is needed and once to actually insert it. That would be a lot slower and I think the tradeoff is acceptable.