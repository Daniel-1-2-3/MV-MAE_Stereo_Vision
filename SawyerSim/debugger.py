class Debugger():
    def __init__(self):
        self.debugs = {
            "fps_renderer": [],
            "fps_encoder_nograd": [],
            "fps_encoder_grad": [],
            "fps_mvmae_grad": [],
            "fps_actor_fwd": [],
            "fps_critics_fwd": []
        }
        
        self.units = {
            "fps_renderer": "ms",
            "fps_encoder_nograd": "ms",
            "fps_encoder_grad": "ms",
            "fps_mvmae_grad": "ms",
            "fps_actor_fwd": "ms",
            "fps_critics_fwd": "ms"
        }
    
    def put(self, value, key):
        self.debugs[key].append(value)
    
    def print_all(self):
        print("")
        for key in self.debugs.keys():
            if len(self.debugs[key]) > 0:
                print(f'{key} (avg {self.units[key]}): {sum(self.debugs[key]) / len(self.debugs[key])}')
    
    def print_one(self, key):
        if len(self.debugs[key]) > 0:
            print(f'{key} (avg {self.units[key]}): {sum(self.debugs[key]) / len(self.debugs[key])}')