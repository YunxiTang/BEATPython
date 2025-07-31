class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        y = x
        x0 = y / 2
        i = 0
        while True:
            x1 = x0 + (y - x0**2) / (2 * x0)
            if abs(x1 - x0) < 1e-2:
                break
            else:
                x0 = x1
                print(f"Iter {i}: {x0}")
                i += 1
        return int(x1)


if __name__ == "__main__":
    import numpy as np

    x = 2147395599
    sol = Solution()
    print(sol.mySqrt(x), np.sqrt(x))
