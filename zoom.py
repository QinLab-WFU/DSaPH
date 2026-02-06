import xlrd
import math
num_bits = 16
num_classes = 10

sheet = xlrd.open_workbook('codetable.xls').sheet_by_index(0)
threshold = sheet.row(num_bits)[math.ceil(math.log(num_classes, 2))].value
