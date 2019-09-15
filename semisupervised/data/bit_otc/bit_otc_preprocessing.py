import sys
import os

e2f = dict()
h2e = dict()
t2e = dict()

valid = list()

fi = open('soc-sign-bitcoinotc.csv', 'r')
for e, line in enumerate(fi):
	items = line.strip().split(',')
	h, t, w = int(items[0]), int(items[1]), int(items[2])
	if w > 3:
		l = 1
	elif w < -3:
		l = 0
	else:
		l = -1
	
	e2f[e] = (h, t, l)
	
	if h not in h2e:
		h2e[h] = set()
	h2e[h].add(e)

	if t not in t2e:
		t2e[t] = set()
	t2e[t].add(e)

	if l != -1:
		valid += [e]
fi.close()

n = len(valid)
n_train = 100 #int(n * 0.01)
n_dev = 500 #int(n * 0.1)
n_test = n - n_train - n_dev
fo = open('train.txt', 'w')
for k in range(n_train):
	fo.write(str(valid[k]) + '\n')
fo.close()
fo = open('dev.txt', 'w')
for k in range(n_train, n_train+n_dev):
	fo.write(str(valid[k]) + '\n')
fo.close()
fo = open('test.txt', 'w')
for k in range(n_train+n_dev, n):
	fo.write(str(valid[k]) + '\n')
fo.close()

fo = open('feature.txt', 'w')
for e, f in e2f.items():
	fo.write(str(e) + '\t' + str(f[0]) + ' ' + str(f[1]) + '\n')
fo.close()

fo = open('label.txt', 'w')
for e, f in e2f.items():
	if f[2] == -1:
		continue
	fo.write(str(e) + '\t' + str(f[2]) + '\n')
fo.close()

fo = open('net.txt', 'w')
for e, f in e2f.items():
	h, t = f[0:2]
	eh = h2e[h]
	et = t2e[t]
	for ee in eh:
		fo.write(str(e) + '\t' + str(ee) + '\t1\n')
	for ee in et:
		fo.write(str(e) + '\t' + str(ee) + '\t1\n')
fo.close()

'''
fo = open('net.txt', 'w')
for e, f in e2f.items():
	h, t = f[0:2]
	eh = h2e[h]
	et = t2e[t]
	for ee in eh:
		fo.write(str(e) + '\t' + str(ee) + '\t1\t0\n')
	for ee in et:
		fo.write(str(e) + '\t' + str(ee) + '\t1\t1\n')
fo.close()
'''

