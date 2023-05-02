from pathlib import Path
import sys
import cv2

def stitch(
    src:Path,
    ext:str=".JPG",
    downscaling_factor:float=.1
) -> str:
    print('(1/3) loading images...')
    imgs = [
        cv2.resize(
            cv2.imread(
                str(img)
            ),
            (0,0),
            fx=downscaling_factor,
            fy=downscaling_factor,
        )
        for img in sorted(src.glob(f"*{ext}"))
    ]

    stitchy=cv2.Stitcher.create()

    print('(2/3) Computing stitch...')
    (status,output)=stitchy.stitch(imgs)
    if status != cv2.STITCHER_OK:
        print("stitching ain't successful", file=sys.stderr)
        raise SystemExit(1)

    print('(3/3) writing to disk...')

    dst = str(src/f"stitch{ext}")
    cv2.imwrite(dst,output)

    return str(src/f"stitch{ext}")

if __name__ == "__main__":
    stitch(Path(sys.argv[1]))
