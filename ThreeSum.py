def three_sum(nums):
    nums.sort()
    sum_list = []
    for i, num in enumerate(nums):
        if i > 0 and num == nums[i - 1]:
            continue
        left_pointer = i + 1
        right_pointer = len(nums) - 1
        while left_pointer < right_pointer:
            sum_num = num + nums[left_pointer] + nums[right_pointer]
            if sum_num > 0:
                right_pointer -= 1
            elif sum_num < 0:
                left_pointer += 1
            else:
                sum_list.append([num, nums[left_pointer], nums[right_pointer]])
                left_pointer += 1
                while nums[left_pointer] == nums[left_pointer - 1] and left_pointer < right_pointer:
                    left_pointer += 1
    return sum_list


if __name__ == "__main__":
    print(three_sum([-1, 0, 1, 2, -1, -4]))