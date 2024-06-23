

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
    
    def create_colorscale(start, end, n_colors):
        def clamp(x):
            return max(0, min(x, 255))

        def rgb_to_hex(r, g, b):
            return f'#{r:02x}{g:02x}{b:02x}'

        color_list = []

        start_r = (start >> 16) & 0xFF
        end_r = (end >> 16) & 0xFF
        decrementor = (start_r - end_r) / (n_colors - 1)

        for i in range(n_colors):
            r = clamp(int(start_r - i * decrementor))
            color_list.append(rgb_to_hex(r, 0, 0))

        return color_list




continuous_nord_color_scale = [
    [0, '#b48ead'], 
    [0.1, 'rgb(46, 52, 64)'],    
    [0.15, 'rgb(59, 66, 82)'],
    [0.3, 'rgb(67, 76, 94)'],
    [0.45, 'rgb(76, 86, 106)'],
    [0.6, 'rgb(94, 129, 172)'],
    [0.75, 'rgb(129, 161, 193)'], 
    [0.9, 'rgb(173, 199, 232)'],  
    [1, 'rgb(224, 236, 255)']     
]

discrete_nord_color_dict = {
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
nord_color_scale = [
    [0, 'rgb(46, 52, 64)'],     # Dark grey
    [0.04, 'rgb(67, 76, 94)'],  # Dark Blue
    [0.08, 'rgb(94, 129, 172)'], # Blue
    [0.12, 'rgb(129, 161, 193)'],# Light Blue
    [0.16, 'rgb(143, 188, 187)'], # Cyan
    [0.2, 'rgb(163, 190, 140)'], # Green
    [0.24, 'rgb(191, 197, 160)'], # Light Green
    [0.28, 'rgb(235, 203, 139)'], # Yellow
    [0.32, 'rgb(208, 135, 112)'], # Orange
    [0.36, 'rgb(191, 97, 106)'],  # Red
    [0.4, 'rgb(255, 255, 255)'],  # White
    [0.5, 'rgb(255, 255, 255)'],  # White
    [0.6, 'rgb(255, 255, 255)'],  # White
    [0.7, 'rgb(255, 255, 255)'],  # White
    [0.8, 'rgb(255, 255, 255)'],  # White
    [0.9, 'rgb(255, 255, 255)'],  # White
    [1, 'rgb(171, 68, 93)']       # Dark Red
]

outlier_color_scale = [
            [0.0, 'rgb(255, 255, 255)'],  # White
            [0.1, 'rgb(255, 255, 255)'],  # White
            [0.2, 'rgb(255, 255, 255)'],  # White
            [0.3, 'rgb(255, 255, 255)'],  # White
            [0.4, 'rgb(255, 255, 255)'],  # White
            [0.5, 'rgb(255, 255, 255)'],  # White
            [0.6, 'rgb(255, 255, 255)'],  # White
            [0.7, 'rgb(255, 255, 255)'],  # White
            [0.8, 'rgb(255, 255, 255)'],  # White
            [0.9, 'rgb(255, 255, 255)'],  # White
            [1.0, 'rgb(255, 0, 0)']       # Red
        ]