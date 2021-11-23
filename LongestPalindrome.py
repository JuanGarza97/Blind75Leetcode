def longest_palindrome(s: str) -> str:
    longest_substring = ""
    for middle_index, char in enumerate(s):
        left_index = middle_index
        right_index = middle_index
        while left_index >= 0 and right_index < len(s) and s[left_index] == s[right_index]:
            if len(longest_substring) < right_index - left_index + 1:
                longest_substring = s[left_index:right_index + 1]
            left_index -= 1
            right_index += 1

        left_index = middle_index
        right_index = middle_index + 1
        while left_index >= 0 and right_index < len(s) and s[left_index] == s[right_index]:
            if len(longest_substring) < right_index - left_index + 1:
                longest_substring = s[left_index:right_index + 1]
            left_index -= 1
            right_index += 1

    return longest_substring


if __name__ == "__main__":
    print(longest_palindrome("babad"))
    print(longest_palindrome("cbbd"))