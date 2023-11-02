
TEST_MODE = 0
TRAIN_MODE = 1
VALID_MODE = 2

def appstr(s, a):
    """Safe appending strings."""
    try:
        return s + a
    except:
        return None