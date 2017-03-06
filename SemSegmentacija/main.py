import os
from tkinter import *
import tensorflow as tf

from PIL import ImageTk, Image, ImageDraw
import numpy as np
from shutil import copyfile, copy

import trainer

pixelButtonPadY = 5
pixelButtonPadX = 10
setPixelButtonPadY = 5
setPixelButtonPadX = 30

DATA_DIR = '/home/ivan/SemSegmentacija/'    #unesite ovdje svoj put do datoteka
TRAINER = None


class Application(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)

        self.parent = master
        # paths to pictures
        self.pathDefaultPictures = os.path.join(DATA_DIR, "Slike/SlikeZaUcenje/")
        self.pathProcessedPictures = os.path.join(DATA_DIR, "Slike/IzlazneSlike/")
        self.pathLabelPictures = os.path.join(DATA_DIR, "Slike/SlikeOznaka/")
        self.pathSegmentation = os.path.join(DATA_DIR, "Slike/")
        self.currentPictureNumber = 1
        self.status = StringVar()

        # lists of picture paths
        self.processedPictures = []
        self.labelPictures = []

        # check if there are any existing label pictures (not a new project)
        for name in os.listdir(self.pathLabelPictures):
            if os.path.isfile(self.pathLabelPictures + name):
                self.labelPictures.append(self.pathLabelPictures + name)

        # get all processed pictures
        for name in os.listdir(self.pathProcessedPictures):
            if os.path.isfile(self.pathProcessedPictures + name):
                self.processedPictures.append(self.pathProcessedPictures + name)

        # updated paths if there are any existing pictures
        self.pathCurrentPicture = self.pathDefaultPictures + "Slika" + str(self.getCurrentPictureNumberString()) + ".png"
        self.pathCurrentLabelPicture = self.pathLabelPictures + "Slika" + self.getCurrentPictureNumberString() + ".png"

        # check if there are processed images
        if len(self.processedPictures) == len(self.labelPictures) and len(self.labelPictures) > 0:
            # ucitaj model
            pass

        self.toolsWindow = Toplevel(self)

        # frame for pixel buttons
        self.pixelButtons = Frame(self.toolsWindow)
        self.pixelButtons.grid(column=0)

        # Radio buttons for pixel selector size
        self.pixelSelectorSize = Frame(self.toolsWindow)
        self.pixelSelectorSize.grid(column=0, row=10)
        self.setUpRadioButtons()
        self.initSelectorRectangle()

        # selector size default 9px
        self.selectorSizeLower = -2
        self.selectorSizeUpper = 3

        self.processingButtons = Frame(self.toolsWindow)
        self.processingButtons.grid(column=1, row=0)

        # starting detectable DetObjects
        self.teamA = DetObject(self.pixelButtons)
        self.teamB = DetObject(self.pixelButtons)
        self.terrain = DetObject(self.pixelButtons)
        self.crowd = DetObject(self.pixelButtons)
        self.otherPersonel = DetObject(self.pixelButtons)
        self.ball = DetObject(self.pixelButtons)

        # setting up buttons
        self.currentRow = 0
        self.currentColumn = 0

        # list of all detectable DetObjects
        self.detectableObjects = []

        self.pack(side=RIGHT, fill=BOTH, expand=1)
        self.button_clicks = 0

        # >>CANVAS<<

        #  canvas width and height
        self.default_width = 0
        self.default_height = 0

        # set up canvas
        self.scale = 1.0
        self.zimg_id = None
        self.zoomcycle = 0
        self.canvas = Canvas(self, highlightthickness=0)

        # set up scrollbar
        hbar = Scrollbar(self, orient=HORIZONTAL, command=self.canvas.xview)
        hbar.pack(side=BOTTOM, fill=X)
        vbar = Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)
        vbar.pack(side=RIGHT, fill=Y)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.configure(scrollregion=(0, 0, 1000, 1000))

        # windows scroll
        #self.canvas.bind_all("<MouseWheel>", self.zoomer)
        self.canvas.bind("<Button-4>", self.zoom_in)
        self.canvas.bind("<Button-5>", self.zoom_out)
        self.canvas.bind("<Motion>", self.crop)
        self.canvas.bind("<Button 1>", self.setPixel)

        self.statusBar = Label(self, textvariable=self.status, bd=0, relief=SUNKEN,
                               anchor=W, font="Helvetica 11 bold")
        # update current picture
        self.refreshCanvas()

        # create menu
        self.menu = Menu(master)
        master.config(menu=self.menu)

        # create submenu
        self.subMenu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.subMenu)
        self.subMenu.add_command(label="Exit", command=exit)

        self.create_widgets()
		
    def initSelectorRectangle(self):
        self.firstPixelPoint = (-1, -1)
        self.secondPixelPoint = (-1, -1)
        self.selectorRectangleXLower = None
        self.selectorRectangleXUpper = None
        self.selectorRectangleYLower = None
        self.selectorRectangleYUpper = None

    def setUpRadioButtons(self):
        MODES = [
            ("1px", "1"),
            ("9px", "2"),
            ("25px", "3"),
            ("49px", "4"),
            ("81px", "5"),
            ("225px", "6")
        ]
        self.v = StringVar()
        self.v.set("3")  # initialize
        Label(self.pixelSelectorSize, text="Pixel selector size:").pack(anchor=N)
        for text, mode in MODES:
            b = Radiobutton(self.pixelSelectorSize, text=text,
                            variable=self.v, value=mode, command=self.setPixelSelectorSize)
            b.pack()
        Button(self.pixelSelectorSize, text="Select rectangle", width=20, command=lambda: self.selectRectangle()).pack(padx=5, pady=10)

    def setPixelSelectorSize(self):
        # set selector size to 1 pixel
        if self.v.get() == '1':
            self.selectorSizeLower = 0
            self.selectorSizeUpper = 1
        # set selector size to 9 pixels
        elif self.v.get() == '2':
            self.selectorSizeLower = -1
            self.selectorSizeUpper = 2
        # set selector size to 25 pixel
        elif self.v.get() == '3':
            self.selectorSizeLower = -2
            self.selectorSizeUpper = 3
        # set selector size to 49 pixel
        elif self.v.get() == '4':
            self.selectorSizeLower = -3
            self.selectorSizeUpper = 4
        # set selector size to 81 pixel
        elif self.v.get() == '5':
            self.selectorSizeLower = -4
            self.selectorSizeUpper = 5
        # set selector size to 225
        elif self.v.get() == '6':
            self.selectorSizeLower = -7
            self.selectorSizeUpper = 8

    def zoomer(self, event):
        if event.delta > 0:
            if self.zoomcycle != 5:
                self.zoomcycle += 1
        elif event.delta < 0:
            if self.zoomcycle != 0:
                self.zoomcycle -= 1
        self.crop(event)

    def zoom_out(self, event):
        if self.zoomcycle != 0:
            self.zoomcycle -= 1
        self.crop(event)

    def zoom_in(self, event):
        if self.zoomcycle != 5:
            self.zoomcycle += 1
        self.crop(event)

    def crop(self, event):
        if self.zimg_id: self.canvas.delete(self.zimg_id)
        if self.zoomcycle != 0:
            tmp = None
            x, y = event.x, event.y
            # x = self.canvas.canvasx(event.x)
            # y = self.canvas.canvasy(event.y)
            if self.zoomcycle == 1:
                tmp = self.image.crop((x - 45, y - 30, x + 45, y + 30))
            elif self.zoomcycle == 2:
                tmp = self.image.crop((x - 30, y - 20, x + 30, y + 20))
            elif self.zoomcycle == 3:
                tmp = self.image.crop((x - 15, y - 10, x + 15, y + 10))
            elif self.zoomcycle == 4:
                tmp = self.image.crop((x - 6, y - 4, x + 6, y + 4))
            elif self.zoomcycle == 5:
                tmp = self.image.crop((x - 3, y - 2, x + 3, y + 2))
            size = 300, 200
            self.zimg = ImageTk.PhotoImage(tmp.resize(size))
            self.zimg_id = self.canvas.create_image(event.x, event.y, image=self.zimg)

    def refreshCanvas(self):
        # set up canvas
        self.updateCurrentPicturePath()
        if os.path.isfile(self.pathCurrentPicture):
            self.image = Image.open(self.pathCurrentPicture)
            self.draw = ImageDraw.Draw(self.image)
        else:
            self.status.set("Picture" + self.pathCurrentPicture + " doesn't exist!")
            sys.exit(1)
        
        photo = ImageTk.PhotoImage(self.image)
        self.default_width = photo.width()
        self.default_height = photo.height()

        # update status bar
        self.statusBar.pack(side=BOTTOM, fill=X)
        self.status.set("Ready")
        self.canvas.config(width=self.default_width, height=self.default_height)
        self.canvas.pack(pady=50, expand=1)
        self.canvas.create_image(0, 0, image=photo, anchor=NW, tags="image")
        self.canvas.image = photo

        # scale image and canvas size to fit the resolution
        self.parent.update()

        self.status.set("Current picture: " + self.pathCurrentPicture)

    def onClickProcess(self):
        # update status
        self.status.set("Provodim segmentaciju...")
        self.statusBar.configure(fg="#006400")

        # 0. provjeri postoji li oznacavanje
        # countLabelPictures = 0
        # for name in os.listdir(self.pathLabelPictures):
        #    if os.path.isfile(self.pathLabelPictures + name):
        #        countLabelPictures += 1
        # for detObject in self.detectableObjects:
        #   countLabelPictures += len(detObject.canvasIds)
        # if countLabelPictures == 0:
        #    self.status.set("Nije postavljena nijedna oznaka!")
        #    self.statusBar.configure(fg="#E50000")
        #    return


        # Copy current picture to done picture folder
        # if os.path.isfile(self.pathCurrentPicture):
        #    copy(self.pathCurrentPicture, self.pathProcessedPictures)
        # else:
        #    self.status.set("Picture does not exist! " + self.pathCurrentPicture + "\nNo more pictures to copy!")
        ## 1. odredi putanju segmentirane slike
        self.currentSegmentationPath = self.pathSegmentation + str(self.currentPictureNumber) + ".png"
        TRAINER.infer(-1)

        # 2. provedi segmentaciju
        # self.model.obradi(
        #  self.putanjaTekuceSlike,
        #  self.putanjaTekuceSegmentacije)

        # 3. javi GUI-ju da osvjezi prikaz
        # ...

    def onClickAnnotationDone(self):
        # 1. spremi tekucu sliku oznaka na disk, ponisti oznake
        markList = []
        for detObject in self.detectableObjects:
            pixelList = []
            for pixels in detObject.pixels:
                for pixel in pixels:
                    pixelList.append(pixel[1] * self.default_width + pixel[0])
            markList.append(pixelList)

        # Make a byte image of markings
        #list = []
        #for i in range(self.default_height * self.default_width):
        #    classNumber = 0
        #    added = False
        #    for detObjectPixels in markList:
        #        if i in detObjectPixels:
        #            list.append(classNumber)
        #            added = True
        #        classNumber += 1
        #    if not added:
        #        list.append(255)
				
		# improved algorithm
        list = [255] * (self.default_height * self.default_width)
        classNumber = 0
        for detObjectPixels in markList:
            for pixel in detObjectPixels:
                list[pixel] = classNumber
            classNumber += 30

        my_array = np.array(list)
        my_array = my_array.reshape((self.default_height, self.default_width))
        my_array = my_array.reshape((self.default_height, self.default_width)).astype('uint8')
        im = Image.fromarray(my_array)
        # im = im.resize((self.default_width, self.default_height), Image.NEAREST)
        im.save(self.pathCurrentLabelPicture)

        # im.save("output2.png")

        # Copy current picture to done picture folder
        if os.path.isfile(self.pathCurrentPicture):
            copy(self.pathCurrentPicture, self.pathProcessedPictures)
        else:
            self.status.set("Picture does not exist! " + self.pathCurrentPicture + "\nNo more pictures to copy!")
        # Clear canvas (remove markings)
        for detObject in self.detectableObjects:
            detObject.clearAllPixels(self.canvas)

        # 2. azuriraj strukture
        self.processedPictures.append(self.pathCurrentPicture)
        self.labelPictures.append(self.pathCurrentLabelPicture)

        TRAINER.add_image(self.pathCurrentPicture.split('/')[-1])

        self.nextPictureNumber()
        self.updateCurrentLabelPicturePath()
        self.updateCurrentPicturePath()
        self.refreshCanvas()


        # print("ODAVDE")
        # print(self.processedPictures)
        # print(self.labelPictures)
        # print(self.pathCurrentPicture)
        # print(self.pathCurrentLabelPicture)

        # 3. nauci model
        # self.model=semseg.stvori_model(
        #  self.slike_za_ucenje, self.slike_oznaka)

        # 4. obradi tekucu sliku
        # self.onClickProcess()

        # 5. spremi model

        # 6. putanje slika u konfiguracijskoj datoteci

    def repeatEpoch(self):
        TRAINER.train(1)
    
    def loadWeights(self):
        TRAINER.loadWeights()
    
    def saveWeights(self):
        TRAINER.saveWeights()
    
    def getCurrentPictureNumberString(self):
        return str(self.currentPictureNumber)

    def nextPictureNumber(self):
        self.currentPictureNumber += 1

    # set single pixel
    def toggleSetPixel(self, detObject):
        detObject.toggleSet = True
        self.setCursor()

    # set all pixels, repeatingly
    def toggleSetAllPixels(self, detObject):
        if detObject.toggleSetAll:
            self.resetSetButtons(detObject)
            self.resetCursor()
        else:
            # modify button
            detObject.setPixelsButton.configure(text="Done", relief="sunken")
            # reset all other buttons
            for detObjectCheck in self.detectableObjects:
                if detObjectCheck.toggleSetAll:
                    self.resetSetButtons(detObjectCheck)
            detObject.toggleSetAll = True
            self.toggleSetPixel(detObject)

    def setCursor(self):
        self.configure(cursor="circle")
        self.canvas.configure(cursor="circle")

    def resetCursor(self):
        self.configure(cursor="arrow")
        self.canvas.configure(cursor="arrow")

    def setPixel(self, event):
        for detObject in self.detectableObjects:
            if detObject.toggleSet:
                # first check if pixel selector rectangle is selected
                if self.firstPixelPoint[0] != -1 and self.secondPixelPoint[0] != -1:
                    # draw on canvas
                    rect = self.canvas.create_rectangle((self.firstPixelPoint[0]),
                                                        (self.firstPixelPoint[1]),
                                                        (self.secondPixelPoint[0]),
                                                        (self.secondPixelPoint[1]),
                                                        fill=detObject.color, outline=detObject.color, width=1)
                    rectDraw = self.draw.rectangle([(self.firstPixelPoint[0]),
                                                    (self.firstPixelPoint[1]),
                                                    (self.secondPixelPoint[0]),
                                                    (self.secondPixelPoint[1])],
                                                   fill=detObject.color, outline=detObject.color)
                # else normal selector draw
                else:
                    # draw on canvas
                    rect = self.canvas.create_rectangle((event.x + self.selectorSizeLower),
                                                        (event.y + self.selectorSizeLower),
                                                        (event.x + (self.selectorSizeUpper - 1)),
                                                        (event.y + (self.selectorSizeUpper - 1)),
                                                        fill=detObject.color, outline=detObject.color, width=1)
                    rectDraw = self.draw.rectangle([(event.x + self.selectorSizeLower),
                                                    (event.y + self.selectorSizeLower),
                                                    (event.x + (self.selectorSizeUpper - 1)),
                                                    (event.y + (self.selectorSizeUpper - 1))],
                                                   fill=detObject.color, outline=detObject.color)
                # set(add) the pixel internally
                self.setPixelDetObject(event, detObject)
                # save the oval object id for easy removing
                detObject.canvasIds.append(rect)
                self.crop(event)

    def setPixelDetObject(self, event, detObject):
        # outputting x and y coordinates to console
        # print((event.x, event.y))
        pixels = []
        if self.selectorRectangleXLower is not None and self.selectorRectangleYLower is not None:
            for i in range(self.selectorRectangleYLower, self.selectorRectangleYUpper):
                for j in range(self.selectorRectangleXLower, self.selectorRectangleXUpper):
                    pixel = (event[0] + j, event[1] + i)
                    # print "ovaj pixel"
                    # print pixel
                    pixels.append(pixel)
            self.initSelectorRectangle()
            self.canvas.bind("<Button 1>", self.setPixel)
        else:
            for i in range(self.selectorSizeLower, self.selectorSizeUpper):
                for j in range(self.selectorSizeLower, self.selectorSizeUpper):
                    pixel = (event.x + j, event.y + i)
                    pixels.append(pixel)
        # find the right pixel button
        i = 0
        for pixelButton in detObject.pixels:
            if pixelButton is None:
                continue
            i += 1
        # toggle button color
        # detObject.buttons[i].configure(fg="#057c05")
        # add to list
        detObject.pixels.insert(i, pixels)
        # check if only one pixel set or setting all
        if not detObject.toggleSetAll:
            detObject.toggleSet = False
        # if all set do nothing(stop pixel selecting)
        #if i == (len(detObject.buttons) - 1):
        #    self.toggleSetAllPixels(detObject)
        #    return

    def resetPixelButton(self, button, i):
        button.configure(text="Pixel " + str(i + 1), fg="red", relief="raised")

    # reset all pixel setting buttons
    def resetSetButtons(self, detObject):
        detObject.toggleSetAll = False
        detObject.toggleSet = False
        detObject.setPixelsButton.configure(text="Set", relief="raised")

    # get current available position for a new DetObject
    def getPosition(self):
        column = 0
        if len(self.detectableObjects) > 4:
            column = 1
        row = self.currentRow
        return row, column

    def create_widgets(self):

        # TEAM A
        self.teamA.name = "Team A"
        self.teamA.color = "red"
        self.teamA.createButtons(self.canvas, self.getPosition())
        self.teamA.setPixelsButton["command"] = lambda: self.toggleSetAllPixels(self.teamA)
        # add to all detectable object list
        self.detectableObjects.append(self.teamA)
        self.updateRowColumn()

        # TEAM B
        self.teamB.name = "Team B"
        self.teamB.createButtons(self.canvas, self.getPosition())
        self.teamB.setPixelsButton["command"] = lambda: self.toggleSetAllPixels(self.teamB)
        # set starting values
        self.teamB.color = "blue"
        # add to all detectable object list
        self.detectableObjects.append(self.teamB)
        self.updateRowColumn()

        # TERRAIN
        self.terrain.name = "Terrain"
        self.terrain.createButtons(self.canvas, self.getPosition())
        self.terrain.setPixelsButton["command"] = lambda: self.toggleSetAllPixels(self.terrain)
        # set starting values
        self.terrain.color = "yellow"
        # add to all detectable object list
        self.detectableObjects.append(self.terrain)
        self.updateRowColumn()

        # CROWD
        self.crowd.name = "Crowd"
        self.crowd.createButtons(self.canvas, self.getPosition())
        self.crowd.setPixelsButton["command"] = lambda: self.toggleSetAllPixels(self.crowd)
        # set starting values
        self.crowd.color = "grey"
        # add to all detectable object list
        self.detectableObjects.append(self.crowd)
        self.updateRowColumn()
	
        # Other personel
        self.otherPersonel.name = "Other personel"
        self.otherPersonel.createButtons(self.canvas, self.getPosition())
        self.otherPersonel.setPixelsButton["command"] = lambda: self.toggleSetAllPixels(self.otherPersonel)
        # set starting values
        self.otherPersonel.color = "green"
        # add to all detectable object list
        self.detectableObjects.append(self.otherPersonel)
        self.updateRowColumn()

        # Ball
        self.ball.name = "Ball"
        self.ball.createButtons(self.canvas, self.getPosition())
        self.ball.setPixelsButton["command"] = lambda: self.toggleSetAllPixels(self.ball)
        # set starting values
        self.ball.color = "purple"
        # add to all detectable object list
        self.detectableObjects.append(self.ball)
        self.updateRowColumn()

	# Clear all buttom		
        buttonClear = Button(self.pixelButtons, text="Clear All", width=10, command=lambda: self.clearCanvas()).grid(pady=20, padx=5)

        # Annotation done button
        self.annotationDone = Button(self.processingButtons, text="Annotation done", width=30, bg="#48dfff",
                                     command=lambda: self.onClickAnnotationDone()).grid(
            pady=20, padx=5, column=1, row=13)

        # Segmentation button
        self.segmentationButton = Button(self.processingButtons, text="Segmentation", width=30, bg="#ec4646",
                                         command=lambda: self.onClickProcess()).grid(
            pady=20, padx=5, column=1, row=14)

        # Repeat epoch button
        self.repeatEpochButton = Button(self.processingButtons, text="Repeat epoch", width=30, bg="#90EE90",
                                        command=lambda: self.repeatEpoch()).grid(
            pady=20, padx=5, column=1, row=15)

        # Show progress button
        self.showProgressButton = Button(self.processingButtons, text="Show progress", width=40, height=3, bg="#d3d3d3",
                                         command=lambda: self.create_window()).grid(
            pady=20, padx=5, column=1, row=16)

        # Save Weights button
        self.saveW = Button(self.processingButtons, text="Save weights", width=40, height=3, bg="#d3d3d3",
                                         command=lambda: self.saveWeights()).grid(
            pady=20, padx=5, column=1, row=17)

        # Load Weights button
        self.loadW = Button(self.processingButtons, text="Load weigths", width=40, height=3, bg="#d3d3d3",
                                         command=lambda: self.loadWeights()).grid(
            pady=20, padx=5, column=1, row=18)

    def create_window(self):
        t = Toplevel(self)
        t.wm_title("Window")
        # open image progress
        image = Image.open("Slike/SlikeOznaka/Slika1.png")
        photo = ImageTk.PhotoImage(image)

        # show image progress
        l = Label(t, image=photo)
        l.image = photo
        l.pack(side="top", fill="both", expand=True, padx=100, pady=100)

    def updateCurrentPicturePath(self):
        self.pathCurrentPicture = self.pathDefaultPictures + "Slika" + self.getCurrentPictureNumberString() + ".png"

    def updateCurrentLabelPicturePath(self):
        self.pathCurrentLabelPicture = self.pathLabelPictures + "Slika" + self.getCurrentPictureNumberString() + ".png"

    def updateRowColumn(self):
        self.currentRow += 2
        if self.currentRow > 9:
            self.currentRow = 0
            self.currentColumn += 1

    def selectRectangle(self):
        self.canvas.bind("<Button 1>", self.selectRectanglePoints)

    def selectRectanglePoints(self, event):
        if self.firstPixelPoint[0] == -1 or self.firstPixelPoint[1] == -1:
            self.firstPixelPoint = (event.x, event.y)
        else:
            self.secondPixelPoint = (event.x, event.y)
            midpoint = self.getMidpoint()

            if self.firstPixelPoint[0] > self.secondPixelPoint[0] and self.firstPixelPoint[1] > self.secondPixelPoint[
                1]:
                # switch
                tmpFirst = (self.firstPixelPoint[0], self.firstPixelPoint[1])
                self.firstPixelPoint = (self.secondPixelPoint[0], self.secondPixelPoint[1])
                self.secondPixelPoint = (tmpFirst[0], tmpFirst[1])
                self.selectorRectangleXLower = self.firstPixelPoint[0] - midpoint[0]
                self.selectorRectangleXUpper = self.secondPixelPoint[0] - midpoint[0]
                self.selectorRectangleYLower = midpoint[1] - self.secondPixelPoint[1]
                self.selectorRectangleYUpper = midpoint[1] - self.firstPixelPoint[1]
            elif self.firstPixelPoint[0] > self.secondPixelPoint[0]:
                self.selectorRectangleXLower = midpoint[0] - self.firstPixelPoint[0]
                self.selectorRectangleXUpper = midpoint[0] - self.secondPixelPoint[0]
                self.selectorRectangleYLower = midpoint[1] - self.secondPixelPoint[1]
                self.selectorRectangleYUpper = midpoint[1] - self.firstPixelPoint[1]
            elif self.firstPixelPoint[1] > self.secondPixelPoint[1]:
                self.selectorRectangleXLower = self.firstPixelPoint[0] - midpoint[0]
                self.selectorRectangleXUpper = self.secondPixelPoint[0] - midpoint[0]
                self.selectorRectangleYLower = self.secondPixelPoint[1] - midpoint[1]
                self.selectorRectangleYUpper = self.firstPixelPoint[1] - midpoint[1]
            else:
                self.selectorRectangleXLower = self.firstPixelPoint[0] - midpoint[0]
                self.selectorRectangleXUpper = self.secondPixelPoint[0] - midpoint[0]
                self.selectorRectangleYLower = midpoint[1] - self.secondPixelPoint[1]
                self.selectorRectangleYUpper = midpoint[1] - self.firstPixelPoint[1]

            self.setPixel(midpoint)

    def getMidpoint(self):
        midpoint = ((self.firstPixelPoint[0] + self.secondPixelPoint[0]) // 2,
                    (self.firstPixelPoint[1] + self.secondPixelPoint[1]) // 2)
        return midpoint

    def clearCanvas(self):
        for detObject in self.detectableObjects:
            detObject.clearAllPixels(self.canvas)
            self.image = Image.open(self.pathCurrentPicture)
            self.draw = ImageDraw.Draw(self.image)


class DetObject(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.pixels = []
        self.canvasIds = []
        self.toggleSet = False
        self.toggleSetAll = False
        self.name = ""
        self.parent = master
        self.color = ""

        self.buttons = []

    def createButtons(self, canvas, position):
        row = position[0]
        Label(self.parent, text=self.name, fg="blue").grid(row=row, column=position[1], pady=5)
        row += 1
        self.setPixelsButton = Button(self.parent, text="Set", bg="#d3d3d3", width=20)
        self.setPixelsButton.grid(row=row, column=position[1], pady=setPixelButtonPadY, padx=setPixelButtonPadX)
        row += 1
        # i = 0
        # for i in range(10):
        #    text = "Pixel " + str(i + 1)
        #    button = Button(self.parent, text=text, fg="red")
        #    button.grid(pady=pixelButtonPadY, padx=pixelButtonPadX, column=position[1], row=i + 2)
        #    self.buttons.append(button)
        #    print self.buttons

    def clearAllPixels(self, canvas):
        # clear pixel list
        self.pixels = []
        i = 0
        # reset buttons
        #for button in self.buttons:
        #    self.resetPixelButton(button, i)
        #    i += 1
        # clear canvas
        for itemId in self.canvasIds:
            canvas.delete(itemId)
        self.canvasIds = []

    def resetPixelButton(self, button, i):
        button.configure(text="Pixel " + str(i + 1), fg="red", relief="raised")


def main(argv=None):  # pylint: disable=unused-argument
    global TRAINER
    TRAINER = trainer.Trainer()
    root = Tk()
    root.title("Test")
    root.wm_state()
    app = Application(root)
    app.configure()
    root.mainloop()


if __name__ == '__main__':
    tf.app.run()
    del session
