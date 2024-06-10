

class Nord():
    def __init__(self):
        self.nord_colors = {
            'nord0' : '#2e3440', # black
            'nord1' : '#3b4252', # dark grey 
            'nord2' : '#434c5e', # medium grey
            'nord3' : '#4c566a', # grey
            'nord4' : '#d8dee9', # light grey 
            'nord5' : '#e5e9f0', # lighter grey
            'nord6' : '#eceff4', # white
            'nord7' : '#8fbcbb', # greeblue
            'nord8' : '#88c0d0', # light blue
            'nord9' : '#81a1c1', # mediuum blue
            'nord10' : '#5e81ac', # dark blue
            'nord11' : '#bf616a', # red
            'nord12' : '#d08770', # orange
            'nord13' : '#ebcb8b', # yellow
            'nord14' : '#a3be8c', # green
            'nord15' : '#b48ead', # purble
        }

        self.black = '#2e3440' # black
        self.dark_grey = '#3b4252' # dark grey 
        self.medium_grey = '#434c5e' # medium grey
        self.grey = '#4c566a' # grey
        self.light_grey = '#d8dee9' # light grey 
        self.lighter_grey = '#e5e9f0' # lighter grey
        self.white = '#eceff4' # white
        self.greenish_blue = '#8fbcbb' # greeblue
        self.light_blue = '#88c0d0' # light blue
        self.blue = '#81a1c1' # mediuum blue
        self.dark_blue = '#5e81ac' # dark blue
        self.red = '#bf616a' # red
        self.orange = '#d08770' # orange
        self.yellow = '#ebcb8b' # yellow
        self.green = '#a3be8c' # green
        self.purple = '#b48ead' # purble

    def get(self, color):
        return self.nord_colors[color]
