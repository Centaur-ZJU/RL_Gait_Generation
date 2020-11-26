d1 = {"a":1, "d":2}
d2 = {}
d3 = {"c":3}
print(dict(d1, **d2))
print(d1|d3)