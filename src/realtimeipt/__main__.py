
import sys

from realtimeipt import roi

if __name__ == "__main__":
    points = roi.main(*sys.argv[1:])
    print(f"points: {points}")()
