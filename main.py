import cv2
import numpy as np
import os
import sys

class FoamAnnotate:
	def __init__(self, dir = "Dataset", folder = None):
		# self.dataset = self.dir + ".txt"
		self.dir = dir
		self.dirWalk = []
		self.firstLine = True
		self.colorBLUE = [255,0,0]
		self.colorYELLOW = [0,255,255]
		self.colorRED = [0,0,255]
		self.colorGREEN = [0,255,0]
		self.image = None
		self.tmpImage = None
		self.numClick = 0
		self.linesIdx = []
		self.MAX_DIM = 800.0
		self.ratio = 1.0
		self.y1 = -1
		self.y2 = -1
		self.oldY1 = -1
		self.oldY2 = -1
		self.file = None
		self.indexes = None
		self.nframe = str("")

		if folder is None:

			for dirname, dirnames, filenames in os.walk(dir):
				# print path to all subdirectories first.

				for subdirname in dirnames:
					tmp = []
					dirtmp = os.path.join(dirname, subdirname)
					# tmp.append(os.path.join(dirname, subdirname))
					tmp.append(subdirname)
					# print os.path.join(dirname, subdirname)

					for d, dir, f in os.walk(dirtmp):
						for fn in f:
							# tmp.append(os.path.join(d, fn))
							tmp.append(fn)
							# print os.path.join(d, fn)

					self.dirWalk.append(tmp)

				# print path to all filenames.
				# for filename in filenames:
				# 	tmp.append(os.path.join(dirname, filename))
				# 	print(os.path.join(dirname, filename))

		else:
			tmp = []
			tmp.append(folder)
			for dirname, dirnames, filenames in os.walk(dir + "\\" + folder):

				# print path to all filenames.
				for filename in filenames:
					tmp.append(filename)

			self.dirWalk.append(tmp)

			# print "folder"


	def updateCoords(self):
		self.oldY1 = self.y1
		self.oldY2 = self.y2
		self.linesIdx = []
		self.linesIdx.append((self.colorRED,self.oldY1))
		self.linesIdx.append((self.colorBLUE,self.oldY2))

	def onMouse(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.numClick += 1

			if self.firstLine:
				self.tmpImage = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
				self.writeDocImage()
				self.numClick = 0
				# self.linesIdx = []
				self.firstLine = False
				cv2.line(self.tmpImage, (0, y), (self.tmpImage.shape[1] - 1, y),self.colorRED,1)
				# self.linesIdx.append((self.colorRED, y))
				self.y1 = y

				cv2.imshow("Annotation",self.tmpImage)
				# print "Line Up", y
			else:
				self.firstLine = True
				cv2.line(self.tmpImage, (0, y), (self.tmpImage.shape[1] - 1, y),self.colorBLUE,1)
				# self.linesIdx.append((self.colorBLUE,y))
				self.y2 = y
				self.updateCoords()

				cv2.imshow("Annotation", self.tmpImage)
				# print "Line Down", y

	def file_len(self, fname):
		with open(fname) as f:
			len = 0
			for i, l in enumerate(f):
				len = i +1
		return len

	def writeDocImage(self):
		offset = self.tmpImage.shape[1] / 12
		mid = self.tmpImage.shape[1] / 2
		cv2.line(self.tmpImage, (mid - offset, 0), (mid - offset, self.tmpImage.shape[0] - 1), self.colorYELLOW, 1, 4)
		cv2.line(self.tmpImage, (mid + offset, 0), (mid + offset, self.tmpImage.shape[0] - 1), self.colorYELLOW, 1)

		cv2.putText(self.tmpImage, "Tasto SPACE: salva e prossimo frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
					(225, 120, 0))
		cv2.putText(self.tmpImage, "Tasto ESC: uscita (NON premere X per uscire)", (50, 90), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0, 0, 200))
		cv2.putText(self.tmpImage, "In caso di click errato, NON salvare ma ripetere i click dopo il 2", (50, 110),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
		cv2.putText(self.tmpImage, "Possibilita' di tornare indietro tramite backspace", (50, 130),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
		cv2.putText(self.tmpImage, "Attenzione: cliccare prima linea sopra poi linea sotto", (50, 150),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200))

		cv2.putText(self.tmpImage, "Frame numero: " + self.nframe, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 120, 0))

	def checkLines(self):
		if len(self.linesIdx) < 2:
			return False, "Please finish to draw all lines"

		self.linesIdx = (sorted(self.linesIdx, key=lambda k: [k[1], k[0]]))

		if np.array_equal(self.linesIdx[0][0],self.colorBLUE):
			return False, "Red line must be above the Blue line"

		return True, ""

	def annotate_image(self, name):
		self.image = cv2.imread(name, 0)

		if (self.image.shape[1] > self.MAX_DIM):
			self.ratio = self.MAX_DIM / self.image.shape[1]
			self.image = cv2.resize(self.image, (0, 0), fx=self.ratio, fy=self.ratio)
		elif (self.image.shape[0] > self.MAX_DIM):
			self.ratio = self.MAX_DIM / self.image.shape[0]
			self.image = cv2.resize(self.image, (0, 0), fx=self.ratio, fy=self.ratio)

		self.tmpImage = cv2.cvtColor(self.image.copy(),cv2.COLOR_GRAY2BGR)
		# self.linesIdx = []

		cv2.namedWindow("Annotation")
		cv2.setMouseCallback('Annotation', self.onMouse)

		self.writeDocImage()
		if self.oldY1 > 0 and self.oldY2 > 0:
			cv2.line(self.tmpImage, (0, self.oldY1), (self.tmpImage.shape[1] - 1, self.oldY1), self.colorRED, 1)
			cv2.line(self.tmpImage, (0, self.oldY2), (self.tmpImage.shape[1] - 1, self.oldY2), self.colorBLUE, 1)
		cv2.imshow("Annotation", self.tmpImage)

		while (1):
			k = cv2.waitKey(1)

			# key bindings
			if k == 27:  # esc to exit
				return 1
			if k == 32: # space bar
				# print "space bar"
				value, str = self.checkLines()
				if not value:
					# print self.numLines
					print str
					continue
				return 2
			if k == ord('\b'):
				return 3
				# print "back space"

	def expandString(self, value, expansion):
		value = str(value)
		if len(value) < expansion:
			for i in range(0, expansion-len(value)):
				value = "0" + value
		return value

	def Annotate(self):

		for i,v in enumerate(self.dirWalk):
			self.indexes = None

			fname = self.dir + "\\" + v[0] + ".txt"

			if not os.path.exists(fname):
				self.file = open(fname, 'w')
				self.file.close()
			fileLen = self.file_len(fname)
			self.file = None
			if fileLen == 0:
				self.file = open(fname, 'r+')
			else:
				self.file = open(fname, 'r')
				line = self.file.readline()
				# self.indexes = np.array(line.replace("\n", "").split("\t"), dtype=np.int32)
				self.indexes = int(line.replace("\n", ""))
				self.file.close()
				self.file = open(fname, 'r+')

			# if self.indexes is not None:
			# 	if i < self.indexes[0]:
			# 		continue
			# for f_i, f_val in enumerate(v):
			i = -1
			while i < len(v) -1:
				i += 1
				f_i = i
				f_val = v[i]
				if f_i < 1:
					continue
				if self.indexes is not None:
					if f_i <= self.indexes:
						continue

				self.nframe = str(f_i)
				code = self.annotate_image(self.dir + "\\" + v[0] + "\\" + f_val)
				if code == 2:
					# i2 = self.expandString(i,3)
					f_i2 = self.expandString(f_i,6)
					self.file.seek(0)
					# self.file.writelines(str(i2)+"\t"+str(f_i2)+"\n")
					self.file.writelines(str(f_i2) + "\n")
					self.file.seek(0,2)
					self.file.write(str(f_val)+"\t"+str(int(1.0/self.ratio*self.y1))+"\t"+str(int(1.0/self.ratio*self.y2))+"\n")
					self.file.flush()
				elif code == 1:
					self.file.close()
					return

				elif code == 3:
					if i-2 <= 0:
						i = -1
						f_i = i
						self.indexes = None
					else:
						i -= 2
						f_i = i
						# i2 = self.expandString(i, 3)
						f_i2 = self.expandString(f_i, 6)
						self.file.seek(0)
						# self.file.writelines(str(i2)+"\t"+str(f_i2)+"\n")
						self.file.writelines(str(f_i2) + "\n")
						# self.file.seek(0, 2)
						############################
						self.file.seek(0)
						for x in xrange(int(f_i2) + 1):  ### +1 becouse count start from line 2
							self.file.readline()
						self.file.truncate()
						##########################
						self.file.flush()
						self.indexes -= 1

if __name__ == '__main__':

	folder = None

	if len(sys.argv) == 1:
		FoamAnnotate().Annotate()
	elif len(sys.argv) == 2:
		f = sys.argv[1]
		FoamAnnotate(folder=f).Annotate()
	else:
		print "Input Error"