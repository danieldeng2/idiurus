import { getMousePos, moveMouse, mouseClick } from 'robotjs';

//Get the mouse position, retuns an object with x and y. 
var mouse=getMousePos();
console.log("Mouse is at x:" + mouse.x + " y:" + mouse.y);

//Move the mouse down by 100 pixels.
moveMouse(mouse.x,mouse.y+100);

//Left click!
mouseClick();