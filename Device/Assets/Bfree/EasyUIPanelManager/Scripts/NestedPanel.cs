using UnityEngine;
using UnityEngine.UI;

namespace Bfree
{
    public class NestedPanel : MonoBehaviour
    {
        [Tooltip("Back button of nested panel")]
        public Button backButton;
        private void OnEnable()
        {
            EasyUIPanelManager.instance.NestedPanelActivated(backButton);
        }
        private void OnDisable()
        {
            EasyUIPanelManager.instance.NestedPanelDeactivated(backButton);
        }
        private void OnDestroy()
        {
            EasyUIPanelManager.instance.NestedPanelDeactivated(backButton);
        }
        public void ClosePanel()
        {
            if (EasyUIPanelManager.instance != null)
                if (EasyUIPanelManager.instance.panelBehaviour == EasyUIPanelManager.PanelBehaviour.deactivate)
                {
                    gameObject.SetActive(false); //Deactivate panel
                }
                else
                {
                    Destroy(gameObject); ; //Destroy panel
                }
            else
                Debug.LogError("@EasyUIPanel: EasyUIPanelManager is missing, add prefab to your scene.");

        }
    }
}
