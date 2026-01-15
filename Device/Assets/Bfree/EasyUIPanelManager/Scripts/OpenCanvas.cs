using UnityEngine;
using UnityEngine.UI;

namespace Bfree
{
    public class OpenCanvas : MonoBehaviour
    {
        public void ChangeCanvasTo(GameObject canvas)
        {
            EasyUIPanelManager.instance.ChangeCanvas(canvas);
        }
        public void ChangeImageState(Image image)
        {
            EasyUIPanelManager.instance.ChangeNavbarButtonState(image);
        }
    }
}