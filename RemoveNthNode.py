class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def removeNthFromEnd(head, n: int):
    dummy = ListNode(0, head)
    left_pointer = dummy
    right_pointer = head

    while n > 0 and right_pointer:
        right_pointer = right_pointer.next
        n -= 1

    while right_pointer:
        left_pointer = left_pointer.next
        right_pointer = right_pointer.next
    left_pointer.next = left_pointer.next.next
    return dummy.next


if __name__ == "__main__":
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    result = removeNthFromEnd(head, 2)
    while result.next:
        print(result.val, end=", ")
        result = result.next
    print(result.val)

