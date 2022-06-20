# created by me
from . import Solver
from fractions import Fraction


def processMatrix(request):
    # this will process the given matrix
    # will return the solution if the input is in the right format
    # will return the error message otherwise
    rows = request.GET.get('rows', '')
    cols = request.GET.get('cols', '')
    try:
        rows = int(rows)
    except:
        return "Invalid Rows"
    try:
        cols = int(cols)
    except:
        return "Invalid Cols"
    if rows <= 0: return "Invalid Rows"
    if cols <= 0: return "Invalid Cols"

    givenMatrixString = request.GET.get('givenMatrix', 'none').strip()
    print(givenMatrixString)

    givenMatrix = []
    givenMatrixStrings = givenMatrixString.split("\r\n")
    print(givenMatrixStrings)
    for R in givenMatrixStrings:
        try:
            x = list(map(Fraction, R.split()))
            if len(x) != cols or len(x) < 2: return "Invalid Matrix"
            givenMatrix.append(x)
        except:
            return "Invalid Matrix"
    if len(givenMatrix) != rows: return "Invalid Matrix"
    print(givenMatrix)

    ans = Solver.knock_down_the_system(givenMatrix, 'a')

    # now read the data from solution file
    # ans = "<pre>" + ans + "</pre>"
    return ans





