#include "matrix_multiplication.h"
#include <vector>
#include <gtest/gtest.h>


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

// *********************************************************************************




int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
