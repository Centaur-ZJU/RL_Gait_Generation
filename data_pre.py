import xlrd

f = open('./Gait_Winter.csv', 'w', encoding='utf-8', newline='')
data = xlrd.open_workbook('./Gait_Winter.xlsx')
distance = data.sheet_by_index(0)
table = data.sheet_by_index(4)
# f.write('ankle_pos,ankle_vel,knee_pos,knee_vel,hip_pos,hip_vel')
# f.write('\n')
min_row = 8
max_row = 77
for row in range(min_row, max_row):
    f.write(str(table.cell(row, 2).value))
    f.write(','+str(table.cell(row, 3).value))
    f.write(','+str(table.cell(row, 7).value))
    f.write(','+str(table.cell(row, 8).value))
    f.write(','+str(table.cell(row, 11).value))
    f.write(','+str(table.cell(row, 12).value))
    f.write(','+str(distance.cell(row,2).value-distance.cell(row-1,2).value))
    f.write(','+str(distance.cell(row,3).value))
    f.write('\n')
row=max_row
f.write(str(table.cell(row, 2).value))
f.write(','+str(table.cell(row, 3).value))
f.write(','+str(table.cell(row, 7).value))
f.write(','+str(table.cell(row, 8).value))
f.write(','+str(table.cell(row, 11).value))
f.write(','+str(table.cell(row, 12).value))
f.write(','+str(distance.cell(row,2).value-distance.cell(row-1,2).value))
f.write(','+str(distance.cell(row,3).value))
f.close()