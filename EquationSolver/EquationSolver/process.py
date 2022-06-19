# created by me
from . import Solver
from fractions import Fraction


def processMatrix(request):
    # this will process the given matrix
    # will return the solution if the input is in the right format
    # will return the error message otherwise
    rows = request.GET.get('rows', '')
    cols = request.GET.get('cols', '')
    if rows == '': return "Invalid Rows"
    if cols == '': return "Invalid Cols"
    rows = int(rows); cols = int(cols)
    if rows <= 0: return "Invalid Rows"
    if cols <= 0: return "Invalid Cols"

    givenMatrixString = request.GET.get('givenMatrix', 'none').strip()
    print(givenMatrixString)
    if givenMatrixString == 'none': return "Invalid Matrix"

    givenMatrix = []
    # parse the matrix
    givenMatrixStrings = givenMatrixString.split("\r\n")
    print(givenMatrixStrings)
    for R in givenMatrixStrings:
        try:
            x = list(map(Fraction, R.split()))
            if len(x) != cols: return "Invalid Matrix"
            givenMatrix.append(x)
        except:
            return "Invalid Matrix"
    if len(givenMatrix) != rows: return "Invalid Matrix"
    print(givenMatrix)

    Solver.knock_down_the_system(givenMatrix, 'a')

    # now read the data from solution file
    ans = ""
    with open("solution", "r") as f:
        for line in f:
            ans += "<pre>" + line + "</pre>"
    return ans





