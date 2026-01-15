using UnityEngine;
namespace Bfree
{
    // Can be used to open nested panels in active canvas (EasyUIPanelManager.instance.activeCanvas)
    // 
    // If EasyUIPanelManager.PanelBehaviour is set to "destroy" it will Instantiate your prefabs to active canvases
    // If EasyUIPanelManager.PanelBehaviour is set to "deactivate" it will activate the nested panel gameobject in the scene

    public class PanelSpawner : MonoBehaviour
    {
        public void OpenPanel(GameObject panel)
        {
            if (EasyUIPanelManager.instance != null)
                if (EasyUIPanelManager.instance.panelBehaviour == EasyUIPanelManager.PanelBehaviour.deactivate)
                {
                    panel.SetActive(true); //Activate panel
                }
                else
                {
                    Instantiate(panel, EasyUIPanelManager.instance.activeCanvas.transform); //Spawn panel in active canvas
                }
            else
                Debug.LogError("@EasyUIPanel: EasyUIPanelManager is missing, add prefab to your scene.");
        }
    }
}
