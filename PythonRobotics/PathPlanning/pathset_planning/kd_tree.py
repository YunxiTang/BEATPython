"""KD-Tree"""


class Node(object):
    def __init__(self) -> None:
        self._father = None
        self._left = None
        self._right = None
        self._feature = None
        self._split = None

    @property
    def brother(self):
        """brother node

        Returns:
            _type_: _description_
        """
        if self._father is None:
            ret = None
        else:
            if self._father._left is self:
                ret = self._father.right
            else:
                ret = self._father._left
        return ret

    def __str__(self):
        return f"feature: {str(self._feature)}, split: {str(self._split)}"


class KDTree(object):
    def __init__(self):
        self.root = Node()

    def __str__(self):
        ret = []
        i = 0
        que = [(self.root, -1)]
        while que:
            nd, idx_father = que.pop(0)
            ret.append("%d -> %d: %s" % (idx_father, i, str(nd)))
            if nd.left is not None:
                que.append((nd.left, i))
            if nd.right is not None:
                que.append((nd.right, i))
            i += 1
        return "\n".join(ret)


if __name__ == "__main__":
    node = Node()
    print(node)
