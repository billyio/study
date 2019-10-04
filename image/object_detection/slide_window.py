# sliding window
def sliding_window(img, H_size=32):
    # get shape
    H, W, _ = img.shape
    
    # base rectangle [h, w]
    recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

    # sliding window
    for y in range(0, H, 4):
        for x in range(0, W, 4):
            for rec in recs:
                # get half size of ractangle
                dh = int(rec[0] // 2)
                dw = int(rec[1] // 2)

                # get left top x
                x1 = max(x - dw, 0)
                # get left top y
                x2 = min(x + dw, W)
                # get right bottom x
                y1 = max(y - dh, 0)
                # get right bottom y
                y2 = min(y + dh, H)

                # crop region
                region = img[max(y - dh, 0) : min(y + dh, H), max(x - dw, 0) : min(x + dw, W)]

                # resize crop region
                region = resize(region, H_size, H_size)

                # get HOG feature
                region_hog = HOG(region).ravel()