# created by me
import time
from fractions import Fraction
import copy
from io import StringIO  # Python3 use: from io import StringIO
import sys



def beautiful_row(row):
    output = "("
    for i in row[:-1]:
        output += str(i) + ", "
    output += str(row[-1]) + ")"

    return output


def transpose(matrix):
    n = len(matrix)
    m = len(matrix[0])

    resultant = [[] for _ in range(m)]

    for i in range(n):
        for j in range(m):
            element = matrix[i][j]
            resultant[j].insert(i,element)

    return resultant


def multiply_matrix(matrix_1, matrix_2):
    m = len(matrix_1)     # no of rows in the resultant matrix
    n = len(matrix_2[0])  # no of columns in the resultant matrix
    resultant_matrix = [[] for _ in range(m)]

    for mat_1_row_num in range(len(matrix_1)):
        for mat_2_col_num in range(len(matrix_2[0])):
            element = 0
            for i in range(len(matrix_1[mat_1_row_num])):
                element += matrix_1[mat_1_row_num][i]*matrix_2[i][mat_2_col_num]
            resultant_matrix[mat_1_row_num].append(element)

    return resultant_matrix


def inverse_matrix(given_matrix):

    initial_matrix = copy.deepcopy(given_matrix)
    n = len(initial_matrix)

    # to get the diagonal matrix
    diagonal_matrix = []
    for i in range(n):
        temp_lst = []
        for j in range(n):
            if i == j:
                temp_lst.append(1)
            else:
                temp_lst.append(0)
        diagonal_matrix.append(temp_lst)

    # making the matrix lower triangular
    for x in range(n - 1):
        for i in range(x, n - 1):
            if initial_matrix[x][x] == 0:
                initial_matrix[x], initial_matrix[x + 1] = initial_matrix[x + 1], initial_matrix[x]
                diagonal_matrix[x], diagonal_matrix[x + 1] = diagonal_matrix[x + 1], diagonal_matrix[x]
            l = initial_matrix[i + 1][x] / initial_matrix[x][x]
            l = Fraction(l).limit_denominator()
            for k in range(n):
                initial_matrix[i + 1][k] -= l * initial_matrix[x][k]
                diagonal_matrix[i + 1][k] -= l * diagonal_matrix[x][k]
                initial_matrix[i + 1][k] = Fraction(initial_matrix[i + 1][k]).limit_denominator()
                diagonal_matrix[i + 1][k] = Fraction(diagonal_matrix[i + 1][k]).limit_denominator()

    for i in initial_matrix[0]:
        i = Fraction(i)
    for i in diagonal_matrix[0]:
        i = Fraction(i)

    # making the matrix identity
    for x in range(n - 1, 0, -1):
        for i in range(x - 1, -1, -1):
            l = initial_matrix[i][x] / initial_matrix[x][x]
            l = Fraction(l).limit_denominator()

            for k in range(n):
                initial_matrix[i][k] -= l * initial_matrix[x][k]
                diagonal_matrix[i][k] -= l * diagonal_matrix[x][k]
                initial_matrix[i][k] = Fraction(initial_matrix[i][k]).limit_denominator()
                diagonal_matrix[i][k] = Fraction(diagonal_matrix[i][k]).limit_denominator()

    # making diagonal elements of given matrix to 1
    for i in range(n):
        if initial_matrix[i][i] != 1:
            for j in range(n):
                diagonal_matrix[i][j] = diagonal_matrix[i][j] / initial_matrix[i][i]
            initial_matrix[i][i] = 1

    return diagonal_matrix



def knock_down_the_system(given_matrix, typ):
    import sys
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()



    # examine mystdout.getvalue()
    max_length = 0
    save_repetition = []
    n = len(given_matrix)
    n1 = len(given_matrix[0])

#   getting the initial matrix
    initial_matrix = copy.deepcopy(given_matrix)

#   to get the identity matrix corresponding to the given matrix
    identity_matrix = []
    for i in range(n):
        temp = []
        for j in range(n):
            if i==j:
                temp.append(1)
            else:
                temp.append(0)
        identity_matrix.append(temp)

#   a function to print the matrix
    def print_matrix(given_matrix):
        nonlocal max_length
        n = len(given_matrix)
        n1 = len(given_matrix[0])
        for i in range(n):
            for j in range(n1):
                if len(str(given_matrix[i][j])) > max_length:
                    max_length = len(str(given_matrix[i][j]))


        for u in range(n):
            for v in range(n1):
                given_matrix[u][v] = str(given_matrix[u][v])
                print(format(given_matrix[u][v], f"<{max_length + 4}"), end="")
                given_matrix[u][v] = Fraction(given_matrix[u][v])
            print(format("|", f"<{max_length + 4}"), end="")
            for v in range(n):
                identity_matrix[u][v] = str(identity_matrix[u][v])
                print(format(identity_matrix[u][v], f"<{max_length + 4}"), end="")
                identity_matrix[u][v] = Fraction(identity_matrix[u][v])
            print()
        print("-" * ((max_length + 4) * ((2*n1+1) // 2)) + "*" + "-" * ((max_length + 4) * ((2*n+1) // 2)))
        print()

#   a function to sharpen the matrix
    def sharpen_the_matrix(given_matrix):
        for i in range(len(given_matrix)):
            if set(given_matrix[i]) != {0,}:
                x = min(abs(float(i)) for i in (set(given_matrix[i]) - {0,}))
                bool = True
                loop_passed = False
                factor = x
                int_check = []
                for a in range(n1):
                    if given_matrix[i][a] % factor == 0:
                        int_check.append(given_matrix[i][a]/ factor)
                        bool = True
                    else:
                        bool = False
                        break
                    loop_passed = True
                if bool == True and loop_passed == True and factor>1:
                    print(f"R{i + 1} ---> R{i + 1}/({Fraction(factor).limit_denominator()})")
                    for z in range(n1):
                        given_matrix[i][z] = given_matrix[i][z] / factor
                        given_matrix[i][z] = Fraction(given_matrix[i][z]).limit_denominator()
                    for z in range(n):
                        identity_matrix[i][z] = identity_matrix[i][z] / factor
                        identity_matrix[i][z] = Fraction(identity_matrix[i][z]).limit_denominator()
                    print_matrix(given_matrix)

    # removing the minus from last row of the matrix
    def remove_last_minus(given_matrix):
        # making the first element of last row positive
        for i in range(len(given_matrix)):
            m = 0
            factor_sign = 0
            while m < n1:
                factor_sign = given_matrix[i][m]
                if factor_sign not in [0,]:
                    break
                else:
                    m += 1
            if factor_sign not in [0,1]:
                new_factor = Fraction(1 / (factor_sign)).limit_denominator()
                print(f"R{i+1} ---> ({new_factor})R{i+1}")
                for k in range(n1):
                    given_matrix[i][k] = (new_factor)*given_matrix[i][k]
                for k in range(n):
                    identity_matrix[i][k] = (new_factor)*identity_matrix[i][k]

                print_matrix(given_matrix)

    print()

    print("Your given matrix is: ")
    print_matrix(given_matrix)
    sharpen_the_matrix(given_matrix)

#   making the matrix upper triangular
    for x in range(n-1):
        for i in range(x, n-1):

            if x+1<n:
                p = 0
                q = 0
                while p < n1:
                    if given_matrix[x][p] == 0:
                        p += 1
                    else:
                        break
                while q < n1:
                    if given_matrix[x + 1][q] == 0:
                        q += 1
                    else:
                        break
                if q < p:
                    bool = True
                else:
                    bool = False

                if set(given_matrix[x+1])!={0,} and bool:
                    if save_repetition == []:
                        print(f"R{x+1} <---> R{x+2}")
                        save_repetition.append(f"R{x+1} <---> R{x+2}")
                        given_matrix[x], given_matrix[x + 1] =  given_matrix[x + 1] ,given_matrix[x]
                        identity_matrix[x], identity_matrix[x + 1] =  identity_matrix[x + 1] ,identity_matrix[x]
                        print_matrix(given_matrix)
                        sharpen_the_matrix(given_matrix)
                    else:
                        if save_repetition[-1]!= f"R{x+1} <---> R{x+2}" and bool:
                            save_repetition.append(f"R{x + 1} <---> R{x + 2}")
                            given_matrix[x], given_matrix[x + 1] = given_matrix[x + 1], given_matrix[x]
                            identity_matrix[x], identity_matrix[x + 1] = identity_matrix[x + 1], identity_matrix[x]
                            print_matrix(given_matrix)
                            sharpen_the_matrix(given_matrix)

            if x<min(n,n1):
                if given_matrix[x][x]!=0:
                    l = given_matrix[i+1][x]/given_matrix[x][x]
                    l = Fraction(l).limit_denominator()
                    if l!=0:
                        print(f"R{i+2} ---> R{i+2} - ({l})R{x+1}")
                        for k in range(n1):
                            given_matrix[i+1][k] -= l*given_matrix[x][k]
                            given_matrix[i+1][k] = Fraction(given_matrix[i+1][k]).limit_denominator()
                        for k in range(n):
                            identity_matrix[i+1][k] -= l*identity_matrix[x][k]
                            identity_matrix[i+1][k] = Fraction(identity_matrix[i+1][k]).limit_denominator()
                    if l!=0:
                       print_matrix(given_matrix)
                       sharpen_the_matrix(given_matrix)

    # print("happended at 5")
#   some modification in the matrix
    for i in range(n-1):
        j = 0
        while j < n1-1:
            if given_matrix[i][j] ==0:
                j += 1
            elif j>=0:
                if set(given_matrix[i + 1][:j]) == {0, }:
                    l = given_matrix[i + 1][j] / given_matrix[i][j]
                    if l!=0:
                        print(f"R{i + 2} ---> R{i + 2} - ({l})R{i + 1}")
                        for k in range(j,n1):
                            given_matrix[i + 1][k] -= l*given_matrix[i][k]
                            given_matrix[i + 1][k] = Fraction(given_matrix[i + 1][k]).limit_denominator()
                        for k in range(j,n):
                            identity_matrix[i + 1][k] -= l*identity_matrix[i][k]
                            identity_matrix[i + 1][k] = Fraction(identity_matrix[i + 1][k]).limit_denominator()
                        print_matrix(given_matrix)
                break
    # print("happended at 1")
    sharpen_the_matrix(given_matrix)
    # print("happended at 2")
    remove_last_minus(given_matrix)
    # print("happended at 3")
#   printing the echelon form of the matrix
    print("The echelon form is:")
    print_matrix(given_matrix)

#   finding the inverse of the matrix
    for x in range(n-1,0,-1):
        j = 0
        while j <n1:
            if given_matrix[x][j]==0:
                j += 1
            else:
                for z in range(1,x+1):
                    l = given_matrix[x-z][j]/given_matrix[x][j]
                    if l!=0:
                        print(f"R{x - z + 1} ---> R{x - z + 1} - ({l})R{x + 1}")
                        for i in range(j,n1):
                            given_matrix[x-z][i] -= l*given_matrix[x][i]
                            given_matrix[x-z][i] = Fraction(given_matrix[x-z][i]).limit_denominator()
                        for i in range(0,n):
                            identity_matrix[x-z][i] -= l*identity_matrix[x][i]
                            identity_matrix[x-z][i] = Fraction(identity_matrix[x-z][i]).limit_denominator()
                        print_matrix(given_matrix)
                        sharpen_the_matrix(given_matrix)
                break

#   final forced sharpening of the matrix
    for i in range(n):
        m = 0
        bool = False
        while m<n1:
           if given_matrix[i][m]!=0:
               fac = given_matrix[i][m]
               bool = True
               break
           else:
               m += 1
        if  bool:
            if fac > 1:
                for j in range(n1):
                    given_matrix[i][j] = given_matrix[i][j]/fac
                    given_matrix[i][j] = Fraction(given_matrix[i][j]).limit_denominator()
                for j in range(n):
                    identity_matrix[i][j] = identity_matrix[i][j]/fac
                    identity_matrix[i][j] = Fraction(identity_matrix[i][j]).limit_denominator()
                print(f"R{i + 1} ---> R{i + 1}/({Fraction(fac).limit_denominator()})")
                print_matrix(given_matrix)

    sharpen_the_matrix(given_matrix)
    remove_last_minus(given_matrix)

#   printing the reduced form of the matrix
    print("The row-reduced echelon form and the corresponding E matrix to the given matrix: ")
    print_matrix(given_matrix)

#   finding the rank of the matrix
#   the number of pivots in the matrix is equal to the rank(r) of the matrix
#   n1-1-r gives the number of free variables in the matrix,
#   here r is the rank of the matrix and n is the number of columns vectors
    pivots = {}

    rank = 0
    for i in range(n):
        j = 0
        while j < n1:
            if given_matrix[i][j] != 0:
                pivots[i] = j
                j += 1
                rank += 1
                break
            else:
                j += 1

    # these variables will refer to the reverse of the lists in the further calculations
    pivot_col = [i for i in pivots.values()]
    pivot_row = [i for i in pivots.keys()]

    if typ=='a':
        print("The rank of the matrix is:", rank)

    if typ=='n':

        local_rank = 0
        for i in range(n):
            j = 0
            while j < n1:
                if given_matrix[i][j] != 0:
                    pivots[i] = j
                    j += 1
                    local_rank += 1
                    break
                else:
                    j += 1
        print("The rank of the matrix is:", local_rank)
        print("The dimension of nullspace space:", n1-local_rank)

    #     converting the elements to string for printing
        for i in range(len(given_matrix)):
            for j in range(len(given_matrix[i])):
                given_matrix[i][j] = str(given_matrix[i][j])
        for i in range(len(identity_matrix)):
            for j in range(len(identity_matrix[i])):
                identity_matrix[i][j] = str(identity_matrix[i][j])

        print("-----------------------------------------------------------------------------------------")

    #    printing the row space of the given matrix::::::::
        print("1. The dimension of row space:", local_rank)
        print("2. The row space of the following matrix is:")
        print("   * ", end="")
        if not local_rank:
            print("{0}")
        else:
            row_basis = [given_matrix[i] for i in range(local_rank)]
            indp = [f"a{i+1}" for i in range(len(row_basis))]
            rhs_ls = [f"{indp[i]}{beautiful_row(row_basis[i])} + " for i in range(len(row_basis)-1)]
            for i in rhs_ls:
                print(i, end="")
            print(f"{indp[-1]}{beautiful_row(row_basis[-1])}")
            print("   ", end="")
            print("Here a{i} belongs to R.")
        print("-----------------------------------------------------------------------------------------")

    #    printing the column space of the given matrix
        print("1. The dimension of column space:", local_rank)
        print("2. The column space of the following matrix is:")
        print("   * ", end="")
        if not local_rank:
            print("{0}")
        else:
            col_basis = []
            for i in pivot_col:
                temp = []
                for j in range(n):
                    element = initial_matrix[j][i]
                    temp.append(element)
                col_basis.append(temp)

            indp = [f"b{i+1}" for i in range(len(col_basis))]
            rhs_ls = [f"{indp[i]}{beautiful_row(col_basis[i])} + " for i in range(len(col_basis)-1)]
            for i in rhs_ls:
                print(i, end="")
            print(f"{indp[-1]}{beautiful_row(col_basis[-1])}")
            print("   ", end="")
            print("Here b{i} belongs to R.")
        print("-----------------------------------------------------------------------------------------")

    #   printing the left nullspace of the given matrix
        print("1. The dimension of left nullspace space:", n-local_rank)
        print("2. The left nullspace space of the following matrix is:")
        print("   * ", end="")
        if n-local_rank==0:
            print("{0}")
        else:
            lft_nul_basis = []
            for i in range(-1,-(n-local_rank+1),-1):
                lft_nul_basis.append(identity_matrix[i])

            indp = [f"c{i+1}" for i in range(len(lft_nul_basis))]
            rhs_ls = [f"{indp[i]}{beautiful_row(lft_nul_basis[i])} + " for i in range(len(lft_nul_basis)-1)]
            for i in rhs_ls:
                print(i, end="")
            print(f"{indp[-1]}{beautiful_row(lft_nul_basis[-1])}")
            print("   ", end="")
            print("Here c{i} belongs to R.")
        print("-----------------------------------------------------------------------------------------")


    #     converting the elements back to fractions for further calculations
        for i in range(len(given_matrix)):
            for j in range(len(given_matrix[i])):
                given_matrix[i][j] = Fraction(given_matrix[i][j]).limit_denominator()
        for i in range(len(identity_matrix)):
            for j in range(len(identity_matrix[i])):
                identity_matrix[i][j] = Fraction(identity_matrix[i][j]).limit_denominator()

        print("The following is the analysis for the nullspace:")
        print()
    #   adding 0s at the end of each row to calculate the nullspace
        for i in range(len(given_matrix)):
            element = Fraction(0).limit_denominator()
            given_matrix[i].append(element)
        typ="a"
        n1 = len(given_matrix[0])
        n = len(given_matrix)

#   checking for the no solution condition
    if typ == 'a':
        for i in range(n):
            if given_matrix[i][-1]!=0 and set(given_matrix[i][:-1])=={0,}:
                print("The following system of equations has no solution.")
                # # TODO: I have to ask whether to provide the best solution
                # while True:
                #     order = input("Do you want to find the best solution to the system?(yes/no): ").lower()
                #     if order in ['yes', 'no']: break
                #     else:
                #         continue
                # if order=='no':
                #     break
                # else:
                #     # here i will be collecting the first n-1 columns of the initial augmented matrix
                #     A = [[initial_matrix[j][i] for i in range(len(initial_matrix[0])-1)] for j in range(len(initial_matrix))]
                #     b = [[initial_matrix[j][-1]] for j in range(len(initial_matrix))]
                #     A_trns = transpose(A)
                #     A_trns_A = multiply_matrix(A_trns, A)
                #     A_trns_A_inv = inverse_matrix(A_trns_A)
                #     A_trns_A_inv_A_trns = multiply_matrix(A_trns_A_inv, A_trns)
                #     projection = multiply_matrix(A_trns_A_inv_A_trns, b)
                #     print("The following is the best answer to the following system.")
                #     print("(", end=" ")
                #     for i in projection[:-1]:
                #         print(i[0], end=", ")
                #     print(projection[-1][0], end=" ")
                #     print(")")

                sys.stdout = old_stdout
                return mystdout.getvalue()
#   when the rows>=columns(unknowns)
    if typ == 'a' and n >= n1-1:
            if n1-1 == rank: # there is definitely trivial solution
                try:
                    result_list = []
                    last_unknown = given_matrix[rank - 1][rank] / given_matrix[rank - 1][rank - 1]
                    result_list.append(round(last_unknown, 6))

                    for i in range(rank - 1):
                        for j in range(len(result_list)):
                            given_matrix[rank - i - 2][rank] -= result_list[j] * given_matrix[rank - i - 2][rank - j - 1]
                        x = given_matrix[rank - i - 2][rank] / given_matrix[rank - i - 2][rank - 2 - i]
                        result_list.append(round(x, 6))

                    output_lst = result_list[::-1]
                    for i in range(len(output_lst)):
                        output_lst[i] = round(float(output_lst[i]),3)
                    solution_vector = str(output_lst)[1:-1]
                    print(f"The solution space of the given matrix is the vector: ({solution_vector})")
                    # print(f"The dimension of solution space is {n1-1-rank}")
                    print("This is the trivial solution.")
                    # print(f"The no. of independent equ: {rank}")
                    # print(f"The number of dependent equ: {n - rank}")
                    # print(f"The number of free variables: {n1-1-rank}")
                except:
                    print("The following system has no solution.")

            else:   # there are definitely infinite solutions
                free_variables1 = n1-rank-1
                print(f"1. The dimension of the solution is: {free_variables1}")
                pivot_col = [i for i in pivots.values()][::-1]
                pivot_row = [i for i in pivots.keys()][::-1]

                n1 += 1

                # creating the solution matrix
                solution_matrix = [[0 for x in range(n1 - 1)] for y in range(n1 - 1)]

                idpt = []
                # making this change for the trick I found
                for i in range(len(given_matrix)):
                    given_matrix[i].append(0)

                for i in range(n1 - 1):
                    if i not in pivots.values():
                        idpt.append(i)

                for i in idpt:
                    for j in range(n1 - 1):
                        if i != j:
                            solution_matrix[i][j] = 0
                        else:
                            solution_matrix[i][i] = 1

                for i in pivot_col:
                    pivot_col_no = i
                    pivot_col_ind = pivot_col.index(i)
                    pivot_row_no = pivot_row[pivot_col_ind]
                    for j in range(pivot_col_no + 1, n1 - 1):
                        for x in range(j, n1 - 1):

                            if x in idpt:
                                solution_matrix[x][pivot_col_no] = -given_matrix[pivot_row_no][j]/( given_matrix[pivot_row_no][pivot_col_no])

                                break
                            else:

                                solution_matrix[x][pivot_col_no] += -(solution_matrix[x][j]*given_matrix[pivot_row_no][j])/( given_matrix[pivot_row_no][pivot_col_no])

                for i in range(len(solution_matrix)):
                    solution_matrix[i].pop(-1)



                new_solution_matrix = []
                i = 0
                while i < n1 - 1:
                    if i in idpt:
                        new_solution_matrix.append(solution_matrix[i])
                    i += 1
                for i in range(n1 - 2):
                    new_solution_matrix[-1][i] = (-1) * new_solution_matrix[-1][i]

                # printing the free variables
                free = [f'x{i+1}' for i in idpt[:-1]]
                print("2. The free variables are:", beautiful_row(free))
                i = 0
                print("3. The basis for the following system is: ")
                for i in range(len(new_solution_matrix )):
                    for j in range(len(new_solution_matrix [i])):
                        new_solution_matrix [i][j] = float((new_solution_matrix [i][j]))
                        if float(str(new_solution_matrix [i][j])) == int(float((new_solution_matrix [i][j]))):
                            new_solution_matrix [i][j] = int(float((new_solution_matrix [i][j])))

                # changing the elements of new solution matrix to fractions
                for s in range(len(new_solution_matrix)):
                    for t in range(len(new_solution_matrix[s])):
                        new_solution_matrix[s][t] = str(Fraction(new_solution_matrix[s][t]).limit_denominator())

                # printing the basis vectors
                for i in range(len(new_solution_matrix) - 1):
                    print("   ",beautiful_row(new_solution_matrix[i]))

                print("4. The pointing vector to the solution space is:")
                print("   ", beautiful_row(new_solution_matrix[-1]))

                print("5. Hence, the complete solution is following: ")
                print("(*) ", end="")
                variables_list = [f'x{i + 1}' for i in range(n1 - 2)]
                rhs_ls = [f" + {free[i]}{beautiful_row(new_solution_matrix[i])}" for i in range(len(new_solution_matrix) - 1)]
                print(f"{beautiful_row(variables_list)} = {beautiful_row(new_solution_matrix[-1])}", end="")
                for i in rhs_ls:
                    print(i, end="")
                print()

#   when the rows<columns(unknowns)
    elif typ=='a' and n<n1-1:
        free_variables1 = n1 - rank - 1
        print(f"1. The dimension of the solution is: {free_variables1}")
        pivot_col = [i for i in pivots.values()][::-1]
        pivot_row = [i for i in pivots.keys()][::-1]
        # if given_matrix[pivot_row[0]][pivot_col[0]] != 0 and given_matrix[pivot_row[0]][-1] == 0:
        #     print("The following system has no solutions.")
        # else:
        n1 += 1

        # creating the solution matrix
        solution_matrix = [[0 for x in range(n1 - 1)] for y in range(n1 - 1)]

        idpt = []
        # making this change for the trick I found
        for i in range(len(given_matrix)):
            given_matrix[i].append(0)

        for i in range(n1 - 1):
            if i not in pivots.values():
                idpt.append(i)

        for i in idpt:
            for j in range(n1 - 1):
                if i != j:
                    solution_matrix[i][j] = 0
                else:
                    solution_matrix[i][i] = 1

        for i in pivot_col:
            pivot_col_no = i
            pivot_col_ind = pivot_col.index(i)
            pivot_row_no = pivot_row[pivot_col_ind]
            for j in range(pivot_col_no + 1, n1 - 1):
                for x in range(j, n1 - 1):

                    if x in idpt:
                        solution_matrix[x][pivot_col_no] = -given_matrix[pivot_row_no][j] / (given_matrix[pivot_row_no][pivot_col_no])

                        break
                    else:

                        solution_matrix[x][pivot_col_no] += -(solution_matrix[x][j] * given_matrix[pivot_row_no][j]) / (given_matrix[pivot_row_no][pivot_col_no])

        for i in range(len(solution_matrix)):
            solution_matrix[i].pop(-1)



        new_solution_matrix = []
        i = 0
        while i < n1 - 1:
            if i in idpt:
                new_solution_matrix.append(solution_matrix[i])
            i += 1
        for i in range(n1 - 2):
            new_solution_matrix[-1][i] = (-1) * new_solution_matrix[-1][i]

        # printing the free variables
        free = [f'x{i+1}' for i in idpt[:-1]]
        print("2. The free variables are:", beautiful_row(free))

        i = 0
        print("3. The basis for the following system is: ")
        for i in range(len(new_solution_matrix)):
            for j in range(len(new_solution_matrix[i])):
                new_solution_matrix[i][j] = float((new_solution_matrix[i][j]))
                if float((new_solution_matrix[i][j])) == int(float((new_solution_matrix[i][j]))):
                    new_solution_matrix[i][j] = int(float((new_solution_matrix[i][j])))

        # changing the elements of new solution matrix to fractions
        for s in range(len(new_solution_matrix)):
            for t in range(len(new_solution_matrix[s])):
                new_solution_matrix[s][t] = str(Fraction(new_solution_matrix[s][t]).limit_denominator())

        # printing the basis vectors
        for i in range(len(new_solution_matrix) - 1):
            print("   ",beautiful_row(new_solution_matrix[i]))

        print("4. The pointing vector to the solution space is:")
        print("   ", beautiful_row(new_solution_matrix[-1]))
        print("5. Hence, the complete solution is following: ")
        print("(*) ", end="")
        variables_list = [f'x{i+1}' for i in range(n1-2)]
        rhs_ls = [f" + {free[i]}{beautiful_row(new_solution_matrix[i])}" for i in range(len(new_solution_matrix)-1)]
        print(f"{beautiful_row(variables_list)} = {beautiful_row(new_solution_matrix[-1])}", end = "")
        for i in rhs_ls:
            print(i, end = "")
        print()

    sys.stdout = old_stdout
    return mystdout.getvalue()