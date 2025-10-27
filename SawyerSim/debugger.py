class Debugger():
    def __init__(self):
        self.debugs = None
        self.clear()
        
        self.units = {
            "speed_one_renderer": "ms",
            "speed_both_renderer": "ms",
            "speed_encoder_grad": "ms",
            "speed_mvmae_grad": "ms",
            "speed_actor_fwd": "ms",
            "speed_critics_fwd": "ms"
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
            
    def clear(self):
        self.debugs = {
            "speed_one_renderer": [],
            "speed_both_renderer": [],
            "speed_encoder_nograd": [],
            "speed_encoder_grad": [],
            "speed_mvmae_grad": [],
            "speed_actor_fwd": [],
            "speed_critics_fwd": []
        }