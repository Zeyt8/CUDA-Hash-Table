#ifndef _HASHCPU_
#define _HASHCPU_

#include <stdint.h>

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		~GpuHashTable();

	private:
		unsigned int fnvHash(const char* str);
		struct HashTablePair
		{
			uint32_t key;
			uint32_t value;
		};
		HashTablePair **table;
		unsigned int size;
		unsigned int count;
};

#endif
