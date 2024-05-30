#include "matrix_multiplication.h"
#include <vector>
#include <gtest/gtest.h>
#include <random>


// ######################### Source code of multiplyMatrices in src/matrix_mult




void multiplyMatricesWithoutErrors(const std::vector<std::vector<int>> &A,
                                   const std::vector<std::vector<int>> &B,
                                   std::vector<std::vector<int>> &C, int rowsA, int colsA,
                                   int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// TEST ON UNITARY MATRICES ********************************************************

TEST(UnitaryMatricesTests, SignTestPP) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Sign Test failed on + +";
}

TEST(UnitaryMatricesTests, SignTestMP) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {-1}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Sign Test failed on - +";
}

TEST(UnitaryMatricesTests, SignTestPM) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1}
    };
    std::vector<std::vector<int>> B = {
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Sign Test failed on + -";
}

TEST(UnitaryMatricesTests, SignTestMM) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {-1}
    };
    std::vector<std::vector<int>> B = {
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Sign Test failed on - -";
}

TEST(UnitaryMatricesTests, ZeroMatTest) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {0}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Sign Test failed 0 * n";
}

TEST(UnitaryMatricesTests, ZeroVetTest) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1}
    };
    std::vector<std::vector<int>> B = {
            {0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Sign Test failed on n * 0";
}

TEST(UnitaryMatricesTests, ZeroMatVetTest) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {-1}
    };
    std::vector<std::vector<int>> B = {
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Sign Test failed on 0 * 0";
}

TEST(UnitaryMatricesTests, IdentityTest) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1}
    };
    std::vector<std::vector<int>> B = {
            {5}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Sign Test failed on identity";
}



// TEST ON square MATRICES times vector ********************************************************

TEST(SquareVector, ZeroMatTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {0, 0},
            {0, 0}
    };
    std::vector<std::vector<int>> B = {
            {1},
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Zero Mat Test failed";
}


TEST(SquareVector, ZeroVetTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1, 1},
            {1, 1}
    };
    std::vector<std::vector<int>> B = {
            {0},
            {0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Zero Vet Test failed";
}

TEST(SquareVector, IdentityTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1, 0},
            {0, 1}
    };
    std::vector<std::vector<int>> B = {
            {5},
            {5}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Identity Test failed";
}


TEST(SquareVector, IdentityZeroVetTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1, 0},
            {0, 1}
    };
    std::vector<std::vector<int>> B = {
            {0},
            {0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Identity Zero Vet Test failed";
}


TEST(SquareVector, NormalTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1, 2},
            {-3, 7}
    };
    std::vector<std::vector<int>> B = {
            {8},
            {8}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}


TEST(SquareVector, ZeroMatZeroVetTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {0, 0},
            {0, 0}
    };
    std::vector<std::vector<int>> B = {
            {0},
            {0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Zero Mat Zero Vet Test failed";
}



TEST(SquareVector, SquaretimesOneVector) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1, 2},
            {-3, 7}
    };
    std::vector<std::vector<int>> B = {
            {1},
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Matrix times 1 vector failed";
}


TEST(SquareVector, SquareTimesMinusOneVector) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {8, 8},
            {-3, 7}
    };
    std::vector<std::vector<int>> B = {
            {-1},
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Matrix times -1 vector failed";
}


//TEst on rectangular matrices ********************************************************

TEST(RectangularMatrices, NormalTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {1, 8, 3},
            {4, 5, 8}
    };
    std::vector<std::vector<int>> B = {
            {7, 8},
            {9, 10},
            {11, 12}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected = {
            {58, 64},
            {139, 154}
    };

    ASSERT_EQ(C, expected) << "Normal Test failed";
}


TEST(RectangularMatrices, ZeroMatZeroMatTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {0, 0, 0},
            {0, 0, 0}
    };
    std::vector<std::vector<int>> B = {
            {0, 0},
            {0, 0},
            {0, 0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected = {
            {0, 0},
            {0, 0}
    };

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Zero Mat Zero Mat Test failed";
}


TEST(RectangularMatrices, ZeroMatNormalMatTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {0, 0, 0},
            {0, 0, 0}
    };
    std::vector<std::vector<int>> B = {
            {1, 2},
            {3, 4},
            {5, 6}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected = {
            {0, 0},
            {0, 0}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Zero Mat Normal Mat Test failed";
}


TEST(RectangularMatrices, NormalMatZeroMatTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {1, 2, 3},
            {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
            {0, 0},
            {0, 0},
            {0, 0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected = {
            {0, 0},
            {0, 0}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Mat Zero Mat Test failed";
}



TEST(RectangularMatrices, MatIdentityMatTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {1, 0},
            {0, 1}
    };
    std::vector<std::vector<int>> B = {
            {17, 6},
            {7, 1},
            {8, 2}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected = {
            {0, 0},
            {0, 0}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Zero Mat Identity Mat Test failed";
}


TEST(RectangularMatrices, IdentityMatZeroMatTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {1, 0},
            {0, 1}
    };
    std::vector<std::vector<int>> B = {
            {0, 0},
            {0, 0},
            {0, 0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected = {
            {0, 0},
            {0, 0}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Identity Mat Zero Mat Test failed";
}





TEST(RectangularMatrices, ZeroMatZeroMatTest2) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {0, 0, 0},
            {0, 0, 0}
    };
    std::vector<std::vector<int>> B = {
            {0, 0},
            {0, 0},
            {0, 0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected = {
            {0, 0},
            {0, 0}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Zero Mat Zero Mat Test 2 failed";
}

TEST(RectangularVector, NormalTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {6, 6, 6},
            {5, 5, 5}
    };
    std::vector<std::vector<int>> B = {
            {1},
            {1},
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {6},
            {15}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}


TEST(RectangularVector, NormalTest2) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {-33,-33, -33},
            {-16, -1, 0}
    };
    std::vector<std::vector<int>> B = {
            {-1},
            {-1},
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {6},
            {15}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}

// TEST on Vector times 1 Vector ********************************************************

TEST(VectorVector, VectorTimesOneVector) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {33, 33, 33}
    };
    std::vector<std::vector<int>> B = {
            {1},
            {1},
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {18}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}

// TEST on Vector times -1 Vector ********************************************************

TEST(VectorVector, VectorTimesMinusOneVector) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {33, 33, 33}
    };
    std::vector<std::vector<int>> B = {
            {-1},
            {-1},
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {-18}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}

// TEST on Vector times 0 Vector ********************************************************

TEST(VectorVector, VectorTimesZeroVector) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {33, 33, 33}
    };
    std::vector<std::vector<int>> B = {
            {0},
            {0},
            {0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {0}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}

// 0 Vector times 0 Vector

TEST(VectorVector, ZeroVectorTimesZeroVector) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {0, 0, 0}
    };
    std::vector<std::vector<int>> B = {
            {0},
            {0},
            {0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {0}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}


TEST(VectorVector, ZeroVectorTimesNormalVector) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {0, 0, 0}
    };
    std::vector<std::vector<int>> B = {
            {1},
            {11},
            {14}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {0}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}


TEST(VectorVector, NormalVectorTimesNormalVector) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1, 2, 3}
    };
    std::vector<std::vector<int>> B = {
            {1},
            {11},
            {14}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {53}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Normal Test failed";
}

//TEST on Vector times 1 scalar ********************************************************

TEST(ScalarVector, ScalarTimesZeroVector) {
    int aRows = 7;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {0},
            {0},
            {0},
            {0},
            {0},
            {0},
            {0}
    };
    std::vector<std::vector<int>> B = {
            {6}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Scalar times 0 vector failed";
}

TEST(ScalarVector, ScalarTimesNormalVector) {
    int aRows = 7;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1},
            {2},
            {3},
            {4},
            {5},
            {6},
            {7}
    };
    std::vector<std::vector<int>> B = {
            {3}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Scalar times normal vector failed";
}


TEST(ScalarVector, ZeroScalarTimesNormalVector) {
    int aRows = 7;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1},
            {2},
            {3},
            {4},
            {5},
            {6},
            {7}
    };
    std::vector<std::vector<int>> B = {
            {0}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Zero scalar times normal vector failed";
}



TEST(ScalarVector, OneScalarTimesNormalVector) {
    int aRows = 7;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {1},
            {2},
            {3},
            {4},
            {5},
            {6},
            {7}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "One scalar times normal vector failed";
}

// Fuzzy test ********************************************************

TEST(FuzzyTest, FuzzyTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 2;
    auto result_list = std::vector<std::vector<std::vector<int>>>(90000, std::vector<std::vector<int>>(9, std::vector<int>(9, 0)));
    auto expected_list = std::vector<std::vector<std::vector<int>>>(90000, std::vector<std::vector<int>>(9, std::vector<int>(9, 0)));
        #include <random>
    std::random_device rd;
    for (int i = 0; i < 1000; i++) {
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-100, 100);
        std::vector<std::vector<int>> A(aRows, std::vector<int>(aCols, 0));
        std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols, 0));
        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, result_list[i], aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected_list[i], aRows, aCols, bCols);
    }


    aRows = 1;
    aCols = 1;
    bCols = 1;

    for( int i = 1000 ; i < 2000 ; i++){
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-200, 200);
        std::vector<std::vector<int>> A(aRows, std::vector<int>(aCols, 0));
        std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols, 0));
        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, result_list[i], aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected_list[i], aRows, aCols, bCols);
    }


    aRows = 8;
    aCols = 8;
    bCols = 8;

    for( int i = 2000 ; i < 5000 ; i++){
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-200, 200);
        std::vector<std::vector<int>> A(aRows, std::vector<int>(aCols, 0));
        std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols, 0));
        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, result_list[i], aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected_list[i], aRows, aCols, bCols);
    }
    //let's now fuzzy test random dimensions'

    for (int i = 5000; i < 9000; i++) {
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 9);
        aRows = dis(gen);
        aCols = dis(gen);
        bCols = dis(gen);
        std::vector<std::vector<int>> A(aRows, std::vector<int>(aCols, 0));
        std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols, 0));
        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, result_list[i], aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected_list[i], aRows, aCols, bCols);
    }




    ASSERT_EQ(result_list, expected_list) << "Fuzzy test failed";



}







// *********************************************************************************



int main(int argc, char **argv) {


    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
