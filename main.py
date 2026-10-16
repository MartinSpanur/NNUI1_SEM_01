from gui import MatrixGridGUI

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
    app = MatrixGridGUI()

    # The grid is invisible initially.
    #
    # To initialize and display it, call:
    #
    # matrix = np.array(
    #     [
    #         [0, 0, 1, 1, 0],
    #         [0, 2, 3, 4, 0],
    #         [1, 1, 0, 1, 1],
    #     ],
    #     dtype=np.uint8,
    # )
    # app.set_matrix(matrix)

    app.mainloop()

