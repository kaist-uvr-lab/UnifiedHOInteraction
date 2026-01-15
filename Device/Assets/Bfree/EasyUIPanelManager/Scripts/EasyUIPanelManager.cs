using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Bfree
{
    public class EasyUIPanelManager : MonoBehaviour
    {
        public static EasyUIPanelManager instance = null;

        public enum PanelBehaviour
        {
            deactivate,
            destroy
        }
        [Tooltip("use destroy if you want to spawn/destroy --- use deactivate if you want activate/deactivate")]
        public PanelBehaviour panelBehaviour;

        [Tooltip("Set true if you want to enable back button to go back from navigation history")]
        [SerializeField] bool useBackButton = false;

        private List<GameObject> navigationHistory = new();
        private List<Image> navigationHistoryImages = new();

        [Tooltip("List of buttons of all active nested panels. Used in OnPressBack() method to be used for a back button.")]
        [SerializeField] List<Button> activeNestedPanelBackButtons = new();

        [Tooltip("This should be set to homepage")]
        public GameObject activeCanvas;

        [Header("Canvas Navigation Bar")]

        [Tooltip("This should be set to homepage navbar button image")]
        [SerializeField] Image activeNavbarImage;

        [Tooltip("Color of active state of navigation button")]
        [SerializeField] Color32 activeNavigationButtonColor;

        [Tooltip("Color of inactive state of navigation button")]
        [SerializeField] Color32 inactiveNavigationButtonColor;

        private void Awake()
        {
            instance = this;
        }
        private void Start()
        {
            activeNavbarImage.color = activeNavigationButtonColor;
            navigationHistory.Add(activeCanvas);
            navigationHistoryImages.Add(activeNavbarImage);
        }
        private void Update()
        {
#if UNITY_ANDROID
            if (useBackButton)
            {
                if (Input.GetKey(KeyCode.Escape))
                {
                    OnPressBack();
                }
            }
#elif UNITY_WINDOWS
            if (useBackButton)
            {
                if (Input.GetKey(KeyCode.Escape))
                {
                    OnPressBack();
                }
            }
#endif
        }

        /// <summary>
        /// This can be used with back buttons of devices
        /// </summary>
        public void OnPressBack()
        {
            //If there are any active nested panels in the canvas, close them
            if (activeNestedPanelBackButtons.Count > 0)
            {
                activeNestedPanelBackButtons[^1].onClick.Invoke();
            }
            else
            {
                if (navigationHistory.Count <= 1) //Nothing to go back
                {
                    Debug.Log("@EasyUIPanel: No page history");
                }
                else
                {
                    GoBack();
                }
            }
        }

        #region Topbar actions
        // Called when a nested panel is activated,
        // abling us to manage active nested panels

        public void NestedPanelActivated(Button button)
        {
            activeNestedPanelBackButtons.Add(button);
        }
        public void NestedPanelDeactivated(Button button)
        {
            activeNestedPanelBackButtons.Remove(button);
        }
        #endregion

        /// <summary>
        /// Change active canvas
        /// </summary>
        /// <param name="canvas"></param>
        public void ChangeCanvas(GameObject canvas)
        {
            if (activeCanvas == canvas)
                return;

            while (activeNestedPanelBackButtons.Count > 0) // Close all nested panels if there are any
            {
                OnPressBack();
            }
            CloseCanvas(activeCanvas);
            activeCanvas = canvas;
            OpenCanvas(activeCanvas);
            navigationHistory.Add(activeCanvas);
        }
        public void OpenCanvas(GameObject canvas)
        {
            canvas.SetActive(true);
        }
        public void CloseCanvas(GameObject canvas)
        {
            canvas.SetActive(false);
        }
        void GoBack()
        {
            CloseCanvas(activeCanvas);
            activeCanvas = navigationHistory[navigationHistory.Count - 2];
            navigationHistory.RemoveAt(navigationHistory.Count - 1);
            OpenCanvas(activeCanvas);

            //Set image from history
            activeNavbarImage.color = inactiveNavigationButtonColor;
            activeNavbarImage = navigationHistoryImages[navigationHistoryImages.Count - 2];
            activeNavbarImage.color = activeNavigationButtonColor;
            navigationHistoryImages.RemoveAt(navigationHistoryImages.Count - 1);

        }
        public void ChangeNavbarButtonState(Image image) //Change active navbar to active state
        {
            if (activeNavbarImage != image)
            {
                //Old image
                activeNavbarImage.color = inactiveNavigationButtonColor;

                //New image
                activeNavbarImage = image;
                activeNavbarImage.color = activeNavigationButtonColor;
                navigationHistoryImages.Add(image);
            }
        }
    }
}
