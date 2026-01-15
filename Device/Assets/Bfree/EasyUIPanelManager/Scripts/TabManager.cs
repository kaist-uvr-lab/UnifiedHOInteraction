using UnityEngine;
using UnityEngine.UI;

namespace Bfree
{
    public class TabManager : MonoBehaviour
    {
        [Tooltip("Array of tab buttons from left to right")]
        [SerializeField] Button[] tabButtons;
        [Tooltip("Array of tab panels, same order as tab buttons")]
        [SerializeField] GameObject[] tabPanels;

        [Tooltip("Set the active tab button, this will be the default panel of the tab")]
        [SerializeField] Button activeTabButton;
        private GameObject activeTab;

        [Tooltip("Active color of tab button")]
        [SerializeField] Color32 activeTabColor;
        [Tooltip("Inactive color of tab button")]
        [SerializeField] Color32 inactiveTabColor;
        private void Start()
        {
            if (tabButtons.Length == 0 || tabPanels.Length == 0)
            {
                Debug.Log("@EasyUIPanel: One or more of the tab arrays are empty. Set the arrays in the inspector");
            }
            else if (tabButtons.Length != tabPanels.Length)
            {
                Debug.Log("@EasyUIPanel: Set the tab buttons and tab panels at the inspector in the same order");
            }
            else
            {
                Initialize();
            }

        }
        void Initialize()
        {
            int startPosition = 0; //initial start position of tabs
            for (int i = 0; i < tabButtons.Length; i++)
            {
                if (tabButtons[i] == activeTabButton)
                {
                    startPosition = i;
                    break;
                }
            }
            for (int i = 0; i < tabButtons.Length; i++)
            {
                int tabindex = i; // New int for delegate
                tabButtons[i].onClick.AddListener(delegate { ChangePanel(tabindex); }); //Setting listeners for tabs

                if (i == startPosition) //Close all panels if open, set the correct colors
                {
                    tabPanels[i].SetActive(true);
                    activeTab = tabPanels[i];
                    tabButtons[i].GetComponent<Image>().color = activeTabColor;
                }
                else
                {
                    tabPanels[i].SetActive(false);
                    tabButtons[i].GetComponent<Image>().color = inactiveTabColor;
                }
            }
        }
        void ChangePanel(int index)
        {
            Debug.Log("@EasyUIPanel: tab index " + index);

            if (activeTab == tabPanels[index]) //Same tab clicked
                return;
            else
            {
                activeTab.SetActive(false);
                activeTabButton.GetComponent<Image>().color = inactiveTabColor;
                activeTab = tabPanels[index];
                activeTabButton = tabButtons[index];
                activeTab.SetActive(true);
                activeTabButton.GetComponent<Image>().color = activeTabColor;
            }
        }

    }
}
