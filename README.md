## Magic Wand

***See the demo video here:*** https://www.youtube.com/watch?v=L6RWHaveG0M

### Inspiration

The inspiration comes from Meta Workplace blackboard functionality, as well as the fact that some people need to draw their illustrations during interviews or technical meetings, such as drawing a binary tree structure or a mathematical equation. Or maybe someone who is cooking needs to operate computers or tablets to search for recipes. However, mouse or trackpad are not natural as a drawing tool to humans, and not everyone has a tablet or paper and pen to draw on. Thus, we have thought of creating a new way so that people can draw and navigate their cursor in a natural way with great precision.

### What it does

The Magic Wand is a system-level control app that allows user to control their cursor based on user’s hand gesture. User can move mouse around, do left/right click, scroll, and even take a screenshot using different gestures. The most interactive way is to open up a drawing pad and start drawing. User can also use the cursor in a regular way, such as open a new browser tab or select a file and delete it.

### How we built it

We built the Magic Wand based on the open-source library MediaPipe, which does a great job in detecting hand gestures. The MediaPipe library maps 21 points on each of the hand it detects and we can program the specific patterns of gesture using the positions of those points. For controlling the cursor, we used Pynput to perform all the basic cursor actions.

### How to build


```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install PySide6
```
