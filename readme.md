# Winograd F(2,3)

This tutorial shows how to compute wino_f23:
1. define transform matrix
   ```python
   G_F23 = np.array([
        [ 1.0,  0.0, 0.0 ],
        [ 0.5,  0.5, 0.5 ],
        [ 0.5, -0.5, 0.5 ],
        [ 0.0,  0.0, 1.0 ]])
    Bt_F23 = np.array([
        [ 1.0,  0.0, -1.0,  0.0 ],
        [ 0.0,  1.0,  1.0,  0.0 ],
        [ 0.0, -1.0,  1.0,  0.0 ],
        [ 0.0,  1.0,  0.0, -1.0 ]])
    At_F23 = np.array([
        [ 1.0, 1.0,  1.0,  0.0 ],
        [ 0.0, 1.0, -1.0, -1.0 ]])
   ```
2. compute transformation for input, kernel, output
   ```python
    def trans_kernel(g):
        return np.dot(np.dot(G_F23,g),G_F23.T)
    def trans_input(d):
        return np.dot(np.dot(Bt_F23,d),Bt_F23.T)
    def trans_output(r):
        return np.dot(np.dot(At_F23,r),At_F23.T)
   ```
3. do conv_winof23, conv_direct
   ```python
    def wino_f23(kernel,input):
        tran_inp = trans_input(input)
        tran_ker = trans_kernel(kernel)
        mid = tran_inp * tran_ker
        out = trans_output(mid)
        return out

    def conv_direct(kernel,input):
        out=np.zeros((2,2))
        for h in range(2):
            for w in range(2):
                out[h,w]=np.sum(input[h:h+3,w:w+3]*kernel)
        return out
   ```
4. given one test data, test the results
    ```python
    def test():
        input=np.array([
            [0,1,2,3],
            [4,5,6,7],
            [8,9,10,11],
            [12,13,14,15]
        ])
        kernel=np.array([
        [1,2,1],
        [2,1,0],
        [1,1,2]
        ])
        out_wino = wino_f23(kernel,input)
        print("out_wino:\n",out_wino)
        out_direct= conv_direct(kernel,input)
        print("out_direct:\n",out_direct)
        print("max error: ",np.max(np.abs(out_wino-out_direct)))
    ```

run
> python3 wino_f23.py

you will get 
```
out_wino:
 [[  54.   65.]
 [  98.  109.]]
out_direct:
 [[  54.   65.]
 [  98.  109.]]
max error:  0.0
```