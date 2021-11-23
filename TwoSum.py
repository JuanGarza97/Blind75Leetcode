def two_sum(nums, target):
    hash_sum = {}
    for i, num in enumerate(nums):
        check = target - num
        if check in hash_sum:
            return hash_sum[check], i
        hash_sum[num] = i
    return None


if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))
