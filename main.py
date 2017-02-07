import cv2
import numpy as np
import os

class FoamAnnotate:
	def __init__(self, dir = "..\\Dataset"):
		self.dir = dir
		self.dataset = self.dir + ".txt"
		self.dirWalk = []
		self.indexes = [] # indice 0 indica dir, indice 1 indica file
		self.indexes.append((0,0))
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
		self.ratio = 0
		self.y1 = -1
		self.y2 = -1
		self.oldX = -1
		self.oldY = -1
		self.file = None
		self.indexes = None
		self.nframe = str("")

		for dirname, dirnames, filenames in os.walk(dir):
			# print path to all subdirectories first.

			for subdirname in dirnames:
				tmp = []
				dirtmp = os.path.join(dirname, subdirname)
				tmp.append(os.path.join(dirname, subdirname))
				# print os.path.join(dirname, subdirname)

				for d, dir, f in os.walk(dirtmp):
					for fn in f:
						tmp.append(os.path.join(d, fn))
						# tmp.append(fn)
						# print os.path.join(d, fn)

				self.dirWalk.append(tmp)

			# print path to all filenames.
			# for filename in filenames:
			# 	tmp.append(os.path.join(dirname, filename))
			# 	print(os.path.join(dirname, filename))

	def onMouse(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.numClick += 1

			if self.firstLine:
				self.firstLine = False
				cv2.line(self.tmpImage, (0, y), (self.tmpImage.shape[1] - 1, y),self.colorRED,1)
				self.linesIdx.append((self.colorRED, y))
				self.y1 = y

				cv2.imshow("Annotation",self.tmpImage)
				# print "Line Up", y
			else:
				self.firstLine = True
				cv2.line(self.tmpImage, (0, y), (self.tmpImage.shape[1] - 1, y),self.colorBLUE,1)
				self.linesIdx.append((self.colorBLUE,y))
				self.y2 = y

				cv2.imshow("Annotation", self.tmpImage)
				# print "Line Down", y

			if self.numClick >= 2:
				self.tmpImage = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
				self.writeDocImage()
				self.numClick = 0

			if self.numClick == 2:
				self.linesIdx = []

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
					(255, 120, 0))
		cv2.putText(self.tmpImage, "Tasto ESC: uscita (NON premere X per uscire)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0, 0, 255))
		cv2.putText(self.tmpImage, "In caso di click errato, NON salvare ma ripetere i click dopo il 2", (50, 130),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

		cv2.putText(self.tmpImage, "Frame numero: " + self.nframe, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 120, 0))

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
		self.linesIdx = []

		cv2.namedWindow("Annotation")
		cv2.setMouseCallback('Annotation', self.onMouse)

		self.writeDocImage()
		cv2.imshow("Annotation", self.tmpImage)

		while (1):
			k = cv2.waitKey(1)
			# key bindings
			if k == 27:  # esc to exit
				return False
			if k == 32: # space bar
				# print "space bar"
				value, str = self.checkLines()
				if not value:
					# print self.numLines
					print str
					continue
				return True

	def Annotate(self):
		if not os.path.exists(self.dataset):
			self.file = open(self.dataset, 'w')
			self.file.close()

		fileLen = self.file_len(self.dataset)
		self.file = None

		if fileLen == 0:
			self.file = open(self.dataset, 'w')
		else:
			self.file = open(self.dataset,'r')
			line = self.file.readline()
			self.indexes = np.array(line.replace("\n","").split("\t"),dtype=np.int32)
			self.file.close()
			self.file = open(self.dataset, 'r+')

		for i,v in enumerate(self.dirWalk):
			if self.indexes is not None:
				if i < self.indexes[0]:
					continue
			for f_i, f_val in enumerate(v):
				if f_i < 1:
					continue
				if self.indexes is not None:
					if f_i <= self.indexes[1]:
						continue

				self.nframe = str(f_i)
				if self.annotate_image(f_val):
					self.file.seek(0)
					self.file.write(str(i)+"\t"+str(f_i)+"\n")
					self.file.seek(0,2)
					self.file.write(str(f_val)+"\t"+str(self.y1)+"\t"+str(self.y2)+"\n")
					self.file.flush()
					pass ##todo save data
				else:
					self.file.close()
					return

if __name__ == '__main__':

	FoamAnnotate().Annotate()


