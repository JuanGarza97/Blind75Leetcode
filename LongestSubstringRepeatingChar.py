def solution(s):
    sub_sequence = []
    max_length = 0
    for i, char in enumerate(s):
        while char in sub_sequence:
            sub_sequence.pop(0)
        sub_sequence.append(char)
        max_length = max(max_length, len(sub_sequence))
    return max_length


if __name__ == "__main__":
    print(solution("abcabcbb"))
    print(solution("bbbbb"))